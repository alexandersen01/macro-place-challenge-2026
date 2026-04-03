from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence

import torch

from macro_place.benchmark import Benchmark

from differentiable import (
    boundary_penalty,
    hard_macro_overlap_penalty,
    smooth_congestion_cost,
    smooth_density_cost,
    smooth_wirelength_cost,
)
from gpu_cost import compute_proxy_cost
from legalize import legalize_hard_macros
from net_extract import NetlistTensors


def _build_initializations(
    benchmark: Benchmark,
    device: torch.device,
    num_starts: int,
    seed: int,
    seed_positions: Sequence[torch.Tensor] | None = None,
) -> torch.Tensor:
    base = benchmark.macro_positions.to(device)
    starts: List[torch.Tensor] = []

    if seed_positions:
        for seed_position in seed_positions:
            starts.append(_clamp_to_canvas(seed_position.to(device=device, dtype=base.dtype), benchmark))

    starts.append(base.clone())

    starts.append(_greedy_row_init(benchmark, device))
    starts.append(_spiral_init(benchmark, device))

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    while len(starts) < num_starts:
        jitter = torch.randn(
            benchmark.num_macros,
            2,
            generator=gen,
            device=device,
            dtype=base.dtype,
        )
        scale = max(benchmark.canvas_width, benchmark.canvas_height) * 0.1
        starts.append(_clamp_to_canvas(base + jitter * scale, benchmark))

    stacked = torch.stack(starts[:num_starts], dim=0)
    fixed_mask = benchmark.macro_fixed.to(device)
    stacked[:, fixed_mask] = base[fixed_mask]
    return stacked


def _greedy_row_init(benchmark: Benchmark, device: torch.device) -> torch.Tensor:
    out = benchmark.macro_positions.to(device).clone()
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].to(device)
    order = torch.argsort(-(sizes[:, 0] * sizes[:, 1]))
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    for idx in order.tolist():
        w = float(sizes[idx, 0])
        h = float(sizes[idx, 1])
        if x_cursor + w > benchmark.canvas_width:
            x_cursor = 0.0
            y_cursor += row_height
            row_height = 0.0
        out[idx, 0] = x_cursor + w / 2.0
        out[idx, 1] = y_cursor + h / 2.0
        x_cursor += w
        row_height = max(row_height, h)
    return _clamp_to_canvas(out, benchmark)


def _spiral_init(benchmark: Benchmark, device: torch.device) -> torch.Tensor:
    out = benchmark.macro_positions.to(device).clone()
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].to(device)
    order = torch.argsort(-(sizes[:, 0] * sizes[:, 1]))
    center_x = benchmark.canvas_width / 2.0
    center_y = benchmark.canvas_height / 2.0
    radius_step = max(benchmark.canvas_width, benchmark.canvas_height) / max(n_hard, 1)
    for rank, idx in enumerate(order.tolist()):
        angle = rank * (math.pi * (3.0 - math.sqrt(5.0)))
        radius = radius_step * math.sqrt(rank + 1)
        out[idx, 0] = center_x + radius * math.cos(angle)
        out[idx, 1] = center_y + radius * math.sin(angle)
    return _clamp_to_canvas(out, benchmark)


def _clamp_to_canvas(placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
    sizes = benchmark.macro_sizes.to(placement.device)
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    if placement.dim() == 2:
        x = torch.maximum(placement[:, 0], half_w)
        x = torch.minimum(x, benchmark.canvas_width - half_w)
        y = torch.maximum(placement[:, 1], half_h)
        y = torch.minimum(y, benchmark.canvas_height - half_h)
        return torch.stack([x, y], dim=-1)

    x = torch.maximum(placement[..., 0], half_w.unsqueeze(0))
    x = torch.minimum(x, benchmark.canvas_width - half_w.unsqueeze(0))
    y = torch.maximum(placement[..., 1], half_h.unsqueeze(0))
    y = torch.minimum(y, benchmark.canvas_height - half_h.unsqueeze(0))
    return torch.stack([x, y], dim=-1)


def run_analytical_placement(
    benchmark: Benchmark,
    netlist: NetlistTensors,
    device: torch.device,
    num_starts: int = 8,
    num_iters: int = 600,
    seed: int = 42,
    time_budget_s: float | None = None,
    seed_positions: Sequence[torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    init = _build_initializations(
        benchmark,
        device,
        num_starts=num_starts,
        seed=seed,
        seed_positions=seed_positions,
    )
    param = torch.nn.Parameter(init)
    optimizer = torch.optim.Adam([param], lr=max(benchmark.canvas_width, benchmark.canvas_height) / 200.0)
    fixed_mask = benchmark.macro_fixed.to(device)
    base = benchmark.macro_positions.to(device)
    start_time = time.time()
    executed_steps = 0

    for step in range(num_iters):
        if time_budget_s is not None and step > 0 and (time.time() - start_time) >= time_budget_s:
            break

        executed_steps += 1
        frac = step / max(num_iters - 1, 1)
        gamma = max(benchmark.canvas_width, benchmark.canvas_height) * (
            0.05 * (1.0 - frac) + 0.005 * frac
        )
        overlap_weight = 1.0 + 99.0 * frac

        optimizer.zero_grad()
        clamped = _clamp_to_canvas(param, benchmark)
        fixed_values = base.unsqueeze(0).expand_as(clamped)
        clamped = torch.where(
            fixed_mask.view(1, -1, 1),
            fixed_values,
            clamped,
        )

        wl = smooth_wirelength_cost(clamped, benchmark, netlist, gamma=gamma)
        den = smooth_density_cost(clamped, benchmark, netlist, p=8.0)
        cong = smooth_congestion_cost(clamped, benchmark, netlist, p=8.0)
        overlap = hard_macro_overlap_penalty(clamped, benchmark, beta=2.0)
        bounds = boundary_penalty(clamped, benchmark)

        loss = wl + 0.5 * den + 0.5 * cong + overlap_weight * overlap + 10.0 * bounds
        loss.mean().backward()
        optimizer.step()

        with torch.no_grad():
            projected = _clamp_to_canvas(param, benchmark)
            projected[:, fixed_mask] = base[fixed_mask]
            param.copy_(projected)

    with torch.no_grad():
        final_batch = _clamp_to_canvas(param, benchmark)
        final_batch[:, fixed_mask] = base[fixed_mask]
        exact = compute_proxy_cost(final_batch, benchmark, netlist)
        exact_proxy = exact["proxy_cost"]
        best_idx = int(torch.argmin(exact_proxy).item())
        best = final_batch[best_idx].detach().cpu()
        best = legalize_hard_macros(best, benchmark)

    return {
        "best_placement": best,
        "candidate_batch": final_batch.detach().cpu(),
        "candidate_costs": exact_proxy.detach().cpu(),
        "executed_steps": executed_steps,
    }
