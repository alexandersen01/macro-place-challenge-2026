from __future__ import annotations

import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark

from gpu_cost import (
    compute_density_grid,
    compose_pin_positions,
    ensure_batched,
    ensure_feature_batched,
    unbatch,
)
from net_extract import NetlistTensors


def smooth_wirelength_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    gamma: float,
) -> torch.Tensor:
    pin_pos = compose_pin_positions(placement, benchmark, netlist)
    pin_pos_b, squeezed = ensure_batched(pin_pos)
    pin_indices = netlist.net_pins.clamp_min(0)
    net_pin_pos = pin_pos_b[:, pin_indices]
    mask = netlist.net_mask.unsqueeze(0)

    x = net_pin_pos[..., 0]
    y = net_pin_pos[..., 1]
    neg_inf = torch.full_like(x, -torch.inf)
    x_pos = torch.where(mask, x / gamma, neg_inf)
    x_neg = torch.where(mask, -x / gamma, neg_inf)
    y_pos = torch.where(mask, y / gamma, neg_inf)
    y_neg = torch.where(mask, -y / gamma, neg_inf)

    lse_x = gamma * (torch.logsumexp(x_pos, dim=-1) + torch.logsumexp(x_neg, dim=-1))
    lse_y = gamma * (torch.logsumexp(y_pos, dim=-1) + torch.logsumexp(y_neg, dim=-1))
    denom = (benchmark.canvas_width + benchmark.canvas_height) * netlist.net_cnt
    cost = ((lse_x + lse_y) * netlist.net_weights.unsqueeze(0)).sum(dim=-1) / denom
    return unbatch(cost, squeezed)


def smooth_density_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    p: float = 8.0,
) -> torch.Tensor:
    density = compute_density_grid(placement, benchmark, netlist)
    density_b, squeezed = ensure_feature_batched(density)
    cost = 0.5 * density_b.clamp_min(0).pow(p).mean(dim=-1).pow(1.0 / p)
    return unbatch(cost, squeezed)


def smooth_congestion_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    p: float = 8.0,
) -> torch.Tensor:
    from gpu_cost import compute_congestion_maps

    v_map, h_map = compute_congestion_maps(placement, benchmark, netlist)
    combined = torch.cat(
        [ensure_feature_batched(v_map)[0], ensure_feature_batched(h_map)[0]],
        dim=-1,
    )
    cost = combined.clamp_min(0).pow(p).mean(dim=-1).pow(1.0 / p)
    return unbatch(cost, placement.dim() == 2)


def hard_macro_overlap_penalty(
    placement: torch.Tensor,
    benchmark: Benchmark,
    beta: float = 1.0,
) -> torch.Tensor:
    batched, squeezed = ensure_batched(placement)
    hard_pos = batched[:, : benchmark.num_hard_macros]
    sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].to(batched.device)
    sep_x = (sizes[:, 0].unsqueeze(0) + sizes[:, 0].unsqueeze(1)) / 2.0
    sep_y = (sizes[:, 1].unsqueeze(0) + sizes[:, 1].unsqueeze(1)) / 2.0

    dx = (hard_pos[:, :, 0].unsqueeze(2) - hard_pos[:, :, 0].unsqueeze(1)).abs()
    dy = (hard_pos[:, :, 1].unsqueeze(2) - hard_pos[:, :, 1].unsqueeze(1)).abs()
    overlap_x = F.softplus((sep_x.unsqueeze(0) - dx) * beta) / beta
    overlap_y = F.softplus((sep_y.unsqueeze(0) - dy) * beta) / beta
    penalty = overlap_x * overlap_y

    eye = torch.eye(benchmark.num_hard_macros, device=batched.device, dtype=torch.bool)
    penalty = penalty.masked_fill(eye.unsqueeze(0), 0.0)
    penalty = torch.triu(penalty, diagonal=1).sum(dim=(-1, -2))
    return unbatch(penalty, squeezed)


def boundary_penalty(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    batched, squeezed = ensure_batched(placement)
    sizes = benchmark.macro_sizes.to(batched.device)
    half_w = sizes[:, 0].unsqueeze(0) / 2.0
    half_h = sizes[:, 1].unsqueeze(0) / 2.0

    left = (half_w - batched[..., 0]).clamp_min(0.0)
    right = (batched[..., 0] - (benchmark.canvas_width - half_w)).clamp_min(0.0)
    bottom = (half_h - batched[..., 1]).clamp_min(0.0)
    top = (batched[..., 1] - (benchmark.canvas_height - half_h)).clamp_min(0.0)
    penalty = (left + right + bottom + top).sum(dim=-1)
    return unbatch(penalty, squeezed)
