from __future__ import annotations

from typing import Tuple

import torch

from macro_place.benchmark import Benchmark

from net_extract import NetlistTensors


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_batched(placement: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if placement.dim() == 2:
        return placement.unsqueeze(0), True
    if placement.dim() == 3:
        return placement, False
    raise ValueError(f"Expected placement rank 2 or 3, got shape {tuple(placement.shape)}")


def ensure_feature_batched(value: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if value.dim() == 1:
        return value.unsqueeze(0), True
    if value.dim() == 2:
        return value, False
    raise ValueError(f"Expected value rank 1 or 2, got shape {tuple(value.shape)}")


def ensure_grid_batched(value: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if value.dim() == 2:
        return value.unsqueeze(0), True
    if value.dim() == 3:
        return value, False
    raise ValueError(f"Expected grid rank 2 or 3, got shape {tuple(value.shape)}")


def unbatch(value: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return value.squeeze(0) if squeezed else value


def compose_pin_positions(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> torch.Tensor:
    batched, squeezed = ensure_batched(placement)
    device = batched.device
    port_positions = benchmark.port_positions.to(device)
    all_positions = torch.cat(
        [batched, port_positions.unsqueeze(0).expand(batched.shape[0], -1, -1)],
        dim=1,
    )
    pin_pos = all_positions[:, netlist.pin_parent_idx] + netlist.pin_offsets.unsqueeze(0)
    return unbatch(pin_pos, squeezed)


def compute_wirelength_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> torch.Tensor:
    pin_pos = compose_pin_positions(placement, benchmark, netlist)
    pin_pos_b, squeezed = ensure_batched(pin_pos)

    pin_indices = netlist.net_pins.clamp_min(0)
    net_pin_pos = pin_pos_b[:, pin_indices]
    mask = netlist.net_mask.unsqueeze(0)

    x = net_pin_pos[..., 0]
    y = net_pin_pos[..., 1]
    x_min = torch.where(mask, x, torch.full_like(x, torch.inf)).amin(dim=-1)
    x_max = torch.where(mask, x, torch.full_like(x, -torch.inf)).amax(dim=-1)
    y_min = torch.where(mask, y, torch.full_like(y, torch.inf)).amin(dim=-1)
    y_max = torch.where(mask, y, torch.full_like(y, -torch.inf)).amax(dim=-1)

    hpwl = (x_max - x_min) + (y_max - y_min)
    denom = (benchmark.canvas_width + benchmark.canvas_height) * netlist.net_cnt
    cost = (hpwl * netlist.net_weights.unsqueeze(0)).sum(dim=-1) / denom
    return unbatch(cost, squeezed)


def compute_density_grid(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    chunk_size: int = 128,
) -> torch.Tensor:
    batched, squeezed = ensure_batched(placement)
    device = batched.device
    sizes = benchmark.macro_sizes.to(device)

    xl = batched[..., 0] - sizes[:, 0].unsqueeze(0) / 2.0
    xh = batched[..., 0] + sizes[:, 0].unsqueeze(0) / 2.0
    yl = batched[..., 1] - sizes[:, 1].unsqueeze(0) / 2.0
    yh = batched[..., 1] + sizes[:, 1].unsqueeze(0) / 2.0

    cell_xl = netlist.grid_xl.view(1, 1, -1)
    cell_xh = netlist.grid_xh.view(1, 1, -1)
    cell_yl = netlist.grid_yl.view(1, 1, -1)
    cell_yh = netlist.grid_yh.view(1, 1, -1)

    occupied = torch.zeros(
        (batched.shape[0], netlist.grid_xl.numel()),
        dtype=batched.dtype,
        device=device,
    )

    for start in range(0, benchmark.num_macros, chunk_size):
        end = min(start + chunk_size, benchmark.num_macros)
        overlap_w = (
            torch.minimum(xh[:, start:end].unsqueeze(-1), cell_xh)
            - torch.maximum(xl[:, start:end].unsqueeze(-1), cell_xl)
        ).clamp_min(0.0)
        overlap_h = (
            torch.minimum(yh[:, start:end].unsqueeze(-1), cell_yh)
            - torch.maximum(yl[:, start:end].unsqueeze(-1), cell_yl)
        ).clamp_min(0.0)
        occupied = occupied + (overlap_w * overlap_h).sum(dim=1)

    grid_area = netlist.grid_width * netlist.grid_height
    density = occupied / grid_area
    return unbatch(density, squeezed)


def compute_density_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> torch.Tensor:
    density = compute_density_grid(placement, benchmark, netlist)
    density_b, squeezed = ensure_feature_batched(density)
    total_cells = density_b.shape[-1]

    if total_cells < 10:
        nonzero = density_b > 0
        denom = nonzero.sum(dim=-1).clamp_min(1)
        cost = 0.5 * (density_b * nonzero).sum(dim=-1) / denom
        return unbatch(cost, squeezed)

    topk = max(1, total_cells // 10)
    values, _ = torch.topk(density_b, k=topk, dim=-1)
    cost = 0.5 * values.mean(dim=-1)
    return unbatch(cost, squeezed)


def _rudy_directional_maps(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pin_pos = compose_pin_positions(placement, benchmark, netlist)
    pin_pos_b, _ = ensure_batched(pin_pos)

    pin_indices = netlist.net_pins.clamp_min(0)
    net_pin_pos = pin_pos_b[:, pin_indices]
    mask = netlist.net_mask.unsqueeze(0)

    x = net_pin_pos[..., 0]
    y = net_pin_pos[..., 1]
    x_min = torch.where(mask, x, torch.full_like(x, torch.inf)).amin(dim=-1)
    x_max = torch.where(mask, x, torch.full_like(x, -torch.inf)).amax(dim=-1)
    y_min = torch.where(mask, y, torch.full_like(y, torch.inf)).amin(dim=-1)
    y_max = torch.where(mask, y, torch.full_like(y, -torch.inf)).amax(dim=-1)

    bbox_w = (x_max - x_min).clamp_min(1.0e-6)
    bbox_h = (y_max - y_min).clamp_min(1.0e-6)
    bbox_area = (bbox_w * bbox_h).clamp_min(1.0e-6)

    cell_xl = netlist.grid_xl.view(1, 1, -1)
    cell_xh = netlist.grid_xh.view(1, 1, -1)
    cell_yl = netlist.grid_yl.view(1, 1, -1)
    cell_yh = netlist.grid_yh.view(1, 1, -1)

    overlap_w = (
        torch.minimum(x_max.unsqueeze(-1), cell_xh)
        - torch.maximum(x_min.unsqueeze(-1), cell_xl)
    ).clamp_min(0.0)
    overlap_h = (
        torch.minimum(y_max.unsqueeze(-1), cell_yh)
        - torch.maximum(y_min.unsqueeze(-1), cell_yl)
    ).clamp_min(0.0)
    overlap_area = overlap_w * overlap_h

    weights = netlist.net_weights.view(1, -1, 1)
    v_map = (weights * overlap_area / bbox_area.unsqueeze(-1)).sum(dim=1)
    h_map = (weights * overlap_area / bbox_area.unsqueeze(-1)).sum(dim=1)
    return v_map / netlist.grid_v_routes, h_map / netlist.grid_h_routes


def compute_macro_congestion_maps(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batched, squeezed = ensure_batched(placement)
    device = batched.device
    sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].to(device)
    hard_pos = batched[:, : benchmark.num_hard_macros]

    xl = hard_pos[..., 0] - sizes[:, 0].unsqueeze(0) / 2.0
    xh = hard_pos[..., 0] + sizes[:, 0].unsqueeze(0) / 2.0
    yl = hard_pos[..., 1] - sizes[:, 1].unsqueeze(0) / 2.0
    yh = hard_pos[..., 1] + sizes[:, 1].unsqueeze(0) / 2.0

    cell_xl = netlist.grid_xl.view(1, 1, -1)
    cell_xh = netlist.grid_xh.view(1, 1, -1)
    cell_yl = netlist.grid_yl.view(1, 1, -1)
    cell_yh = netlist.grid_yh.view(1, 1, -1)

    v_map = torch.zeros(
        (batched.shape[0], netlist.grid_xl.numel()),
        dtype=batched.dtype,
        device=device,
    )
    h_map = torch.zeros_like(v_map)

    for start in range(0, benchmark.num_hard_macros, chunk_size):
        end = min(start + chunk_size, benchmark.num_hard_macros)
        overlap_x = (
            torch.minimum(xh[:, start:end].unsqueeze(-1), cell_xh)
            - torch.maximum(xl[:, start:end].unsqueeze(-1), cell_xl)
        ).clamp_min(0.0)
        overlap_y = (
            torch.minimum(yh[:, start:end].unsqueeze(-1), cell_yh)
            - torch.maximum(yl[:, start:end].unsqueeze(-1), cell_yl)
        ).clamp_min(0.0)
        v_map = v_map + overlap_x.sum(dim=1)
        h_map = h_map + overlap_y.sum(dim=1)

    v_map = v_map * (netlist.vrouting_alloc / max(netlist.grid_v_routes, 1.0e-6))
    h_map = h_map * (netlist.hrouting_alloc / max(netlist.grid_h_routes, 1.0e-6))
    return unbatch(v_map, squeezed), unbatch(h_map, squeezed)


def _smooth_directional_map(direction_map: torch.Tensor, axis: int, radius: int) -> torch.Tensor:
    if radius <= 0:
        return direction_map
    batched, squeezed = ensure_grid_batched(direction_map)
    rows = batched.shape[-2]
    cols = batched.shape[-1]
    out = torch.zeros_like(batched)

    if axis == 1:
        for row in range(rows):
            for col in range(cols):
                left = max(0, col - radius)
                right = min(cols - 1, col + radius)
                span = right - left + 1
                out[:, row, left : right + 1] += batched[:, row, col].unsqueeze(-1) / span
    else:
        for row in range(rows):
            for col in range(cols):
                lo = max(0, row - radius)
                hi = min(rows - 1, row + radius)
                span = hi - lo + 1
                out[:, lo : hi + 1, col] += batched[:, row, col].unsqueeze(-1) / span

    return unbatch(out, squeezed)


def compute_congestion_maps(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_net, h_net = _rudy_directional_maps(placement, benchmark, netlist)
    v_macro, h_macro = compute_macro_congestion_maps(placement, benchmark, netlist)

    rows = benchmark.grid_rows
    cols = benchmark.grid_cols
    v_grid = ensure_feature_batched(v_net)[0].view(-1, rows, cols)
    h_grid = ensure_feature_batched(h_net)[0].view(-1, rows, cols)
    v_smooth = _smooth_directional_map(v_grid, axis=1, radius=netlist.smooth_range)
    h_smooth = _smooth_directional_map(h_grid, axis=0, radius=netlist.smooth_range)
    v_smooth_b = ensure_grid_batched(v_smooth)[0].reshape(-1, rows * cols)
    h_smooth_b = ensure_grid_batched(h_smooth)[0].reshape(-1, rows * cols)

    v_total = v_smooth_b + ensure_feature_batched(v_macro)[0]
    h_total = h_smooth_b + ensure_feature_batched(h_macro)[0]

    if placement.dim() == 2:
        return v_total.squeeze(0), h_total.squeeze(0)
    return v_total, h_total


def compute_congestion_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> torch.Tensor:
    v_map, h_map = compute_congestion_maps(placement, benchmark, netlist)
    combined = torch.cat(
        [ensure_feature_batched(v_map)[0], ensure_feature_batched(h_map)[0]],
        dim=-1,
    )
    topk = max(1, int(combined.shape[-1] * 0.05))
    values, _ = torch.topk(combined, k=topk, dim=-1)
    cost = values.mean(dim=-1) * netlist.congestion_scale
    return unbatch(cost, placement.dim() == 2)


def compute_proxy_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
    weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
) -> dict:
    wirelength = compute_wirelength_cost(placement, benchmark, netlist)
    density = compute_density_cost(placement, benchmark, netlist)
    congestion = compute_congestion_cost(placement, benchmark, netlist)
    proxy = weights[0] * wirelength + weights[1] * density + weights[2] * congestion
    return {
        "proxy_cost": proxy,
        "wirelength_cost": wirelength,
        "density_cost": density,
        "congestion_cost": congestion,
    }
