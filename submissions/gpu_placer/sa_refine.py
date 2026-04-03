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


def _check_single_overlap(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    idx: int,
    safety_gap: float = 0.05,
) -> torch.Tensor:
    dx = (positions[:, idx, 0].unsqueeze(1) - positions[:, :, 0]).abs()
    dy = (positions[:, idx, 1].unsqueeze(1) - positions[:, :, 1]).abs()
    sep_x = (sizes[idx, 0] + sizes[:, 0]).unsqueeze(0) / 2.0 + safety_gap
    sep_y = (sizes[idx, 1] + sizes[:, 1]).unsqueeze(0) / 2.0 + safety_gap
    overlaps = (dx < sep_x) & (dy < sep_y)
    overlaps[:, idx] = False
    return overlaps.any(dim=1)


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

        chosen = [random.choice(movable_indices) for _ in range(num_chains)]
        partner = [random.choice(movable_indices) for _ in range(num_chains)]
        for chain_idx in range(num_chains):
            i = chosen[chain_idx]
            if ops[chain_idx] < 0.5:
                shift_scale = temperature * (0.3 + 0.7 * (1.0 - frac))
                proposals[chain_idx, i, 0] += random.gauss(0.0, shift_scale)
                proposals[chain_idx, i, 1] += random.gauss(0.0, shift_scale)
            elif ops[chain_idx] < 0.7:
                j = partner[chain_idx]
                if i != j:
                    tmp = proposals[chain_idx, i].clone()
                    proposals[chain_idx, i] = proposals[chain_idx, j]
                    proposals[chain_idx, j] = tmp
            elif ops[chain_idx] < 0.9:
                neighbors = netlist.macro_adjacency[i]
                if neighbors:
                    weights = netlist.macro_adjacency_weights[i]
                    j = random.choices(neighbors, weights=weights, k=1)[0]
                    alpha = random.uniform(0.05, 0.25)
                    proposals[chain_idx, i] = proposals[chain_idx, i] + alpha * (
                        proposals[chain_idx, j] - proposals[chain_idx, i]
                    )
            else:
                proposals[chain_idx, i, 0] = benchmark.canvas_width - proposals[chain_idx, i, 0]

        proposals = _clamp_chain_positions(proposals, benchmark)
        proposals[:, fixed_mask] = base[fixed_mask]

        invalid = torch.zeros(num_chains, dtype=torch.bool, device=device)
        for idx in set(chosen + partner):
            invalid |= _check_single_overlap(proposals[:, : benchmark.num_hard_macros], sizes, idx)
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
