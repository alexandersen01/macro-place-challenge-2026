from __future__ import annotations

import math
import random
import time
from typing import Optional

import torch

from macro_place.benchmark import Benchmark

from fd_soft import optimize_soft_macros
from gpu_cost import compute_proxy_cost
from net_extract import NetlistTensors


def _check_modified_overlaps(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    modified_indices: torch.Tensor,
    safety_gap: float = 0.0,
) -> torch.Tensor:
    valid = modified_indices >= 0
    if not torch.any(valid):
        return torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)

    gather_idx = modified_indices.clamp_min(0)
    moved_pos = torch.gather(
        positions,
        1,
        gather_idx.unsqueeze(-1).expand(-1, -1, positions.shape[-1]),
    )
    moved_sizes = sizes[gather_idx]
    macro_idx = torch.arange(positions.shape[1], device=positions.device).view(1, 1, -1)

    dx = (moved_pos[..., 0].unsqueeze(-1) - positions[:, None, :, 0]).abs()
    dy = (moved_pos[..., 1].unsqueeze(-1) - positions[:, None, :, 1]).abs()
    sep_x = (moved_sizes[..., 0].unsqueeze(-1) + sizes[:, 0].view(1, 1, -1)) / 2.0 + safety_gap
    sep_y = (moved_sizes[..., 1].unsqueeze(-1) + sizes[:, 1].view(1, 1, -1)) / 2.0 + safety_gap
    overlaps = (
        valid.unsqueeze(-1)
        & (dx < sep_x)
        & (dy < sep_y)
        & (macro_idx != gather_idx.unsqueeze(-1))
    )
    return overlaps.any(dim=(-1, -2))


def _build_neighbor_tensors(
    netlist: NetlistTensors,
    num_hard_macros: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_degree = max((len(neighbors) for neighbors in netlist.macro_adjacency[:num_hard_macros]), default=0)
    if max_degree == 0:
        return (
            torch.zeros((num_hard_macros, 0), dtype=torch.long, device=device),
            torch.zeros((num_hard_macros, 0), dtype=torch.float32, device=device),
            torch.zeros((num_hard_macros,), dtype=torch.long, device=device),
        )

    neighbor_ids = torch.full((num_hard_macros, max_degree), -1, dtype=torch.long, device=device)
    neighbor_weights = torch.zeros((num_hard_macros, max_degree), dtype=torch.float32, device=device)
    neighbor_counts = torch.zeros((num_hard_macros,), dtype=torch.long, device=device)
    for macro_idx in range(num_hard_macros):
        neighbors = netlist.macro_adjacency[macro_idx]
        weights = netlist.macro_adjacency_weights[macro_idx]
        degree = len(neighbors)
        if degree == 0:
            continue
        neighbor_ids[macro_idx, :degree] = torch.tensor(neighbors, dtype=torch.long, device=device)
        neighbor_weights[macro_idx, :degree] = torch.tensor(weights, dtype=torch.float32, device=device)
        neighbor_counts[macro_idx] = degree
    return neighbor_ids, neighbor_weights, neighbor_counts


def _sample_weighted_neighbors(
    macro_indices: torch.Tensor,
    neighbor_ids: torch.Tensor,
    neighbor_weights: torch.Tensor,
    neighbor_counts: torch.Tensor,
) -> torch.Tensor:
    sampled = torch.full_like(macro_indices, -1)
    if neighbor_ids.numel() == 0:
        return sampled

    counts = neighbor_counts[macro_indices]
    valid = counts > 0
    if not torch.any(valid):
        return sampled

    valid_indices = macro_indices[valid]
    weights = neighbor_weights[valid_indices]
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    target = torch.rand((valid_indices.shape[0], 1), device=macro_indices.device) * total
    cumulative = weights.cumsum(dim=-1)
    pick = (cumulative < target).sum(dim=-1).clamp(max=weights.shape[-1] - 1)
    sampled[valid] = neighbor_ids[valid_indices, pick]
    return sampled


def _clamp_chain_positions(positions: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    sizes = benchmark.macro_sizes.to(positions.device)
    half_w = sizes[:, 0].view(1, -1) / 2.0
    half_h = sizes[:, 1].view(1, -1) / 2.0
    positions[..., 0] = positions[..., 0].clamp(half_w, benchmark.canvas_width - half_w)
    positions[..., 1] = positions[..., 1].clamp(half_h, benchmark.canvas_height - half_h)
    return positions


def run_parallel_sa(
    seed_position: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    device: torch.device,
    num_chains: int = 64,
    max_steps: int = 2500,
    time_budget_s: Optional[float] = None,
    seed: int = 42,
) -> torch.Tensor:
    random.seed(seed)
    torch.manual_seed(seed)

    base = seed_position.to(device)
    movable = benchmark.get_movable_mask()[: benchmark.num_hard_macros].cpu()
    movable_indices = torch.where(movable)[0].tolist()
    if not movable_indices:
        return seed_position
    movable_tensor = torch.tensor(movable_indices, dtype=torch.long, device=device)
    neighbor_ids, neighbor_weights, neighbor_counts = _build_neighbor_tensors(
        netlist,
        benchmark.num_hard_macros,
        device,
    )

    positions = base.unsqueeze(0).repeat(num_chains, 1, 1)
    jitter = torch.randn(
        num_chains,
        benchmark.num_hard_macros,
        2,
        device=device,
        dtype=positions.dtype,
    ) * (max(benchmark.canvas_width, benchmark.canvas_height) * 0.01)
    positions[:, : benchmark.num_hard_macros] += jitter
    positions = _clamp_chain_positions(positions, benchmark)
    fixed_mask = benchmark.macro_fixed.to(device)
    positions[:, fixed_mask] = base[fixed_mask]

    sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].to(device)
    current = compute_proxy_cost(positions, benchmark, netlist)["proxy_cost"]
    best = positions.clone()
    best_cost = current.clone()
    stagnant = torch.zeros(num_chains, dtype=torch.long, device=device)
    start_time = time.time()

    for step in range(max_steps):
        if time_budget_s is not None and (time.time() - start_time) > time_budget_s:
            break

        frac = step / max(max_steps - 1, 1)
        temp_start = max(benchmark.canvas_width, benchmark.canvas_height) * 0.05
        temp_end = max(benchmark.canvas_width, benchmark.canvas_height) * 0.001
        temperature = temp_start * (temp_end / temp_start) ** frac

        proposals = positions.clone()
        old_cost = current.clone()
        ops = torch.rand(num_chains, device=device)
        chosen = movable_tensor[torch.randint(movable_tensor.numel(), (num_chains,), device=device)]
        partner = movable_tensor[torch.randint(movable_tensor.numel(), (num_chains,), device=device)]
        modified = torch.full((num_chains, 2), -1, dtype=torch.long, device=device)
        modified[:, 0] = chosen

        shift_mask = ops < 0.5
        if torch.any(shift_mask):
            shift_scale = temperature * (0.3 + 0.7 * (1.0 - frac))
            shift = torch.randn(
                (int(shift_mask.sum().item()), 2),
                device=device,
                dtype=positions.dtype,
            ) * shift_scale
            shift_chains = torch.nonzero(shift_mask, as_tuple=False).squeeze(-1)
            proposals[shift_chains, chosen[shift_mask]] += shift

        swap_mask = (ops >= 0.5) & (ops < 0.7) & (chosen != partner)
        if torch.any(swap_mask):
            swap_chains = torch.nonzero(swap_mask, as_tuple=False).squeeze(-1)
            chosen_swap = chosen[swap_mask]
            partner_swap = partner[swap_mask]
            pos_i = proposals[swap_chains, chosen_swap].clone()
            proposals[swap_chains, chosen_swap] = proposals[swap_chains, partner_swap]
            proposals[swap_chains, partner_swap] = pos_i
            modified[swap_chains, 1] = partner_swap

        pull_mask = (ops >= 0.7) & (ops < 0.9)
        if torch.any(pull_mask):
            pull_chains = torch.nonzero(pull_mask, as_tuple=False).squeeze(-1)
            chosen_pull = chosen[pull_mask]
            sampled_neighbors = _sample_weighted_neighbors(
                chosen_pull,
                neighbor_ids,
                neighbor_weights,
                neighbor_counts,
            )
            valid_pull = sampled_neighbors >= 0
            if torch.any(valid_pull):
                valid_chains = pull_chains[valid_pull]
                chosen_valid = chosen_pull[valid_pull]
                neighbors_valid = sampled_neighbors[valid_pull]
                alpha = torch.empty(
                    (int(valid_pull.sum().item()), 1),
                    device=device,
                    dtype=positions.dtype,
                ).uniform_(0.05, 0.25)
                proposals[valid_chains, chosen_valid] = proposals[valid_chains, chosen_valid] + alpha * (
                    proposals[valid_chains, neighbors_valid] - proposals[valid_chains, chosen_valid]
                )

        flip_mask = ops >= 0.9
        if torch.any(flip_mask):
            flip_chains = torch.nonzero(flip_mask, as_tuple=False).squeeze(-1)
            proposals[flip_chains, chosen[flip_mask], 0] = (
                benchmark.canvas_width - proposals[flip_chains, chosen[flip_mask], 0]
            )

        proposals = _clamp_chain_positions(proposals, benchmark)
        proposals[:, fixed_mask] = base[fixed_mask]

        invalid = _check_modified_overlaps(
            proposals[:, : benchmark.num_hard_macros],
            sizes,
            modified,
        )
        proposals[invalid] = positions[invalid]

        new_cost = compute_proxy_cost(proposals, benchmark, netlist)["proxy_cost"]
        delta = new_cost - old_cost
        accept = (delta <= 0) | (
            torch.rand(num_chains, device=device)
            < torch.exp((-delta / max(temperature, 1.0e-8)).clamp(max=40.0))
        )

        positions[accept] = proposals[accept]
        current = torch.where(accept, new_cost, old_cost)

        improved = current < best_cost
        best[improved] = positions[improved]
        best_cost = torch.minimum(best_cost, current)
        stagnant = torch.where(improved, torch.zeros_like(stagnant), stagnant + 1)

        if (step + 1) % 500 == 0:
            best_idx = int(torch.argmin(best_cost).item())
            refined = optimize_soft_macros(best[best_idx], benchmark, netlist, num_steps=30)
            refined_cost = compute_proxy_cost(refined, benchmark, netlist)["proxy_cost"]
            if float(refined_cost) < float(best_cost[best_idx]):
                best[best_idx] = refined
                best_cost[best_idx] = refined_cost
                positions[best_idx] = refined
                current[best_idx] = refined_cost
            reheat_mask = stagnant > 500
            if reheat_mask.any():
                noise = torch.randn_like(positions[reheat_mask, : benchmark.num_hard_macros]) * (
                    temp_start * 0.03
                )
                positions[reheat_mask, : benchmark.num_hard_macros] += noise
                positions = _clamp_chain_positions(positions, benchmark)
                positions[:, fixed_mask] = base[fixed_mask]
                stagnant[reheat_mask] = 0

    best_idx = int(torch.argmin(best_cost).item())
    return best[best_idx].detach().cpu()
