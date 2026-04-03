from __future__ import annotations

import torch

from macro_place.benchmark import Benchmark

from gpu_cost import compose_pin_positions
from net_extract import NetlistTensors


def optimize_soft_macros(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    num_steps: int = 80,
    attract_factor: float = 0.05,
    repel_factor: float = 0.2,
) -> torch.Tensor:
    if benchmark.num_soft_macros == 0:
        return placement

    device = placement.device
    out = placement.clone()
    soft_start = benchmark.num_hard_macros
    soft_end = benchmark.num_macros
    movable_mask = benchmark.get_movable_mask().to(device)

    for _ in range(num_steps):
        pin_pos = compose_pin_positions(out, benchmark, netlist)
        pin_indices = netlist.net_pins.clamp_min(0)
        net_pin_pos = pin_pos[pin_indices]
        mask = netlist.net_mask

        net_centers = (
            (net_pin_pos * mask.unsqueeze(-1)).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp_min(1)
        )

        forces = torch.zeros_like(out)
        macro_counts = torch.zeros((benchmark.num_macros, 1), device=device, dtype=out.dtype)

        parent_idx = netlist.pin_parent_idx[pin_indices]
        for slot in range(netlist.net_pins.shape[1]):
            valid = mask[:, slot]
            if not valid.any():
                continue
            slot_parent = parent_idx[:, slot][valid]
            slot_macro = slot_parent < benchmark.num_macros
            if not slot_macro.any():
                continue
            macro_ids = slot_parent[slot_macro]
            macro_force = net_centers[valid][slot_macro] - out[macro_ids]
            forces.index_add_(0, macro_ids, macro_force)
            macro_counts.index_add_(
                0,
                macro_ids,
                torch.ones((macro_ids.numel(), 1), device=device, dtype=out.dtype),
            )

        forces = forces / macro_counts.clamp_min(1.0)
        forces[:soft_start] = 0.0
        forces[~movable_mask] = 0.0
        out[soft_start:soft_end] = out[soft_start:soft_end] + attract_factor * forces[soft_start:soft_end]

        soft_sizes = benchmark.macro_sizes[soft_start:soft_end].to(device)
        soft_pos = out[soft_start:soft_end]
        if soft_pos.numel() > 0:
            dx = soft_pos[:, 0].unsqueeze(1) - soft_pos[:, 0].unsqueeze(0)
            dy = soft_pos[:, 1].unsqueeze(1) - soft_pos[:, 1].unsqueeze(0)
            sep_x = (soft_sizes[:, 0].unsqueeze(1) + soft_sizes[:, 0].unsqueeze(0)) / 2.0
            sep_y = (soft_sizes[:, 1].unsqueeze(1) + soft_sizes[:, 1].unsqueeze(0)) / 2.0
            overlap_x = (sep_x - dx.abs()).clamp_min(0.0)
            overlap_y = (sep_y - dy.abs()).clamp_min(0.0)
            overlap = overlap_x * overlap_y
            eye = torch.eye(overlap.shape[0], device=device, dtype=torch.bool)
            overlap = overlap.masked_fill(eye, 0.0)
            repulse_x = torch.sign(dx) * overlap
            repulse_y = torch.sign(dy) * overlap
            repulse = torch.stack([repulse_x.sum(dim=1), repulse_y.sum(dim=1)], dim=-1)
            out[soft_start:soft_end] = out[soft_start:soft_end] + repel_factor * repulse

        half_w = benchmark.macro_sizes[:, 0].to(device) / 2.0
        half_h = benchmark.macro_sizes[:, 1].to(device) / 2.0
        out[:, 0] = out[:, 0].clamp(half_w, benchmark.canvas_width - half_w)
        out[:, 1] = out[:, 1].clamp(half_h, benchmark.canvas_height - half_h)
        out[~movable_mask] = placement[~movable_mask]

    return out
