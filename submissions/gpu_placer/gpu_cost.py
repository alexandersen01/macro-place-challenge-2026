from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

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


def _pin_pos_to_grid_cells(
    pin_pos: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.floor(pin_pos[..., 1] / netlist.grid_height).to(torch.long)
    cols = torch.floor(pin_pos[..., 0] / netlist.grid_width).to(torch.long)
    rows = rows.clamp(0, benchmark.grid_rows - 1)
    cols = cols.clamp(0, benchmark.grid_cols - 1)
    return rows, cols, rows * benchmark.grid_cols + cols


def _scatter_horizontal_segments(
    batch_size: int,
    rows: int,
    cols: int,
    batch_idx: torch.Tensor,
    row_idx: torch.Tensor,
    col_start: torch.Tensor,
    col_end: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    out = weights.new_zeros((batch_size, rows, cols))
    valid = col_end > col_start
    if not torch.any(valid):
        return out

    batch_idx = batch_idx[valid]
    row_idx = row_idx[valid]
    col_start = col_start[valid]
    col_end = col_end[valid]
    weights = weights[valid]

    marks = weights.new_zeros((batch_size, rows, cols + 1))
    flat = marks.view(-1)
    stride = cols + 1
    start_idx = (batch_idx * rows + row_idx) * stride + col_start
    end_idx = (batch_idx * rows + row_idx) * stride + col_end
    flat.scatter_add_(0, start_idx, weights)
    flat.scatter_add_(0, end_idx, -weights)
    return marks[..., :-1].cumsum(dim=-1)


def _scatter_vertical_segments(
    batch_size: int,
    rows: int,
    cols: int,
    batch_idx: torch.Tensor,
    row_start: torch.Tensor,
    row_end: torch.Tensor,
    col_idx: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    out = weights.new_zeros((batch_size, rows, cols))
    valid = row_end > row_start
    if not torch.any(valid):
        return out

    batch_idx = batch_idx[valid]
    row_start = row_start[valid]
    row_end = row_end[valid]
    col_idx = col_idx[valid]
    weights = weights[valid]

    marks = weights.new_zeros((batch_size, rows + 1, cols))
    flat = marks.view(-1)
    start_idx = (batch_idx * (rows + 1) + row_start) * cols + col_idx
    end_idx = (batch_idx * (rows + 1) + row_end) * cols + col_idx
    flat.scatter_add_(0, start_idx, weights)
    flat.scatter_add_(0, end_idx, -weights)
    return marks[:, :-1, :].cumsum(dim=1)


def _accumulate_two_pin_maps(
    src_row: torch.Tensor,
    src_col: torch.Tensor,
    dst_row: torch.Tensor,
    dst_col: torch.Tensor,
    net_ids: torch.Tensor,
    unique_count: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
    rows: int,
    cols: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if src_row.numel() == 0:
        zeros = weights.new_zeros((batch_size, rows, cols))
        return zeros, zeros

    pair_unique_count = unique_count[:, net_ids]
    valid = (pair_unique_count != 3) & ((src_row != dst_row) | (src_col != dst_col))
    if not torch.any(valid):
        zeros = weights.new_zeros((batch_size, rows, cols))
        return zeros, zeros

    batch_idx = torch.arange(batch_size, device=src_row.device).view(-1, 1).expand_as(src_row)
    src_cell = src_row * cols + src_col
    dst_cell = dst_row * cols + dst_col
    cell_count = rows * cols
    pair_key = ((net_ids.view(1, -1) * cell_count + src_cell) * cell_count + dst_cell).to(torch.int64)
    key_space = unique_count.shape[1] * cell_count * cell_count
    global_key = batch_idx.to(torch.int64) * key_space + pair_key

    flat_valid = valid.reshape(-1)
    keys = global_key.reshape(-1)[flat_valid]
    if keys.numel() == 0:
        zeros = weights.new_zeros((batch_size, rows, cols))
        return zeros, zeros

    sort_order = torch.argsort(keys)
    keys = keys[sort_order]
    batch_idx = batch_idx.reshape(-1)[flat_valid][sort_order]
    src_row = src_row.reshape(-1)[flat_valid][sort_order]
    src_col = src_col.reshape(-1)[flat_valid][sort_order]
    dst_row = dst_row.reshape(-1)[flat_valid][sort_order]
    dst_col = dst_col.reshape(-1)[flat_valid][sort_order]
    pair_weights = weights.view(1, -1).expand(batch_size, -1).reshape(-1)[flat_valid][sort_order]

    keep = torch.ones_like(keys, dtype=torch.bool)
    keep[1:] = keys[1:] != keys[:-1]

    batch_idx = batch_idx[keep]
    src_row = src_row[keep]
    src_col = src_col[keep]
    dst_row = dst_row[keep]
    dst_col = dst_col[keep]
    pair_weights = pair_weights[keep]

    h_map = _scatter_horizontal_segments(
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        batch_idx=batch_idx,
        row_idx=src_row,
        col_start=torch.minimum(src_col, dst_col),
        col_end=torch.maximum(src_col, dst_col),
        weights=pair_weights,
    )
    v_map = _scatter_vertical_segments(
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        batch_idx=batch_idx,
        row_start=torch.minimum(src_row, dst_row),
        row_end=torch.maximum(src_row, dst_row),
        col_idx=dst_col,
        weights=pair_weights,
    )
    return v_map, h_map


def _accumulate_three_pin_maps(
    unique_ids: torch.Tensor,
    unique_count: torch.Tensor,
    net_weights: torch.Tensor,
    batch_size: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = unique_count == 3
    if not torch.any(mask):
        zeros = net_weights.new_zeros((batch_size, rows, cols), dtype=dtype)
        return zeros, zeros

    batch_idx, net_idx = torch.nonzero(mask, as_tuple=True)
    cell_ids = unique_ids[batch_idx, net_idx]
    y = torch.div(cell_ids, cols, rounding_mode="floor")
    x = torch.remainder(cell_ids, cols)
    weights = net_weights[net_idx].to(dtype=dtype)

    x_order = torch.argsort(x * rows + y, dim=-1)
    y_x = torch.gather(y, 1, x_order)
    x_x = torch.gather(x, 1, x_order)
    y1, y2, y3 = y_x.unbind(dim=-1)
    x1, x2, x3 = x_x.unbind(dim=-1)

    h_batch = []
    h_row = []
    h_start = []
    h_end = []
    h_weight = []
    v_batch = []
    v_start = []
    v_end = []
    v_col = []
    v_weight = []

    def add_h(case_mask: torch.Tensor, row: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> None:
        idx = torch.nonzero(case_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        h_batch.append(batch_idx[idx])
        h_row.append(row[idx])
        h_start.append(start[idx])
        h_end.append(end[idx])
        h_weight.append(weights[idx])

    def add_v(case_mask: torch.Tensor, start: torch.Tensor, end: torch.Tensor, col: torch.Tensor) -> None:
        idx = torch.nonzero(case_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return
        v_batch.append(batch_idx[idx])
        v_start.append(start[idx])
        v_end.append(end[idx])
        v_col.append(col[idx])
        v_weight.append(weights[idx])

    l_case = (x1 < x2) & (x2 < x3) & (torch.minimum(y1, y3) < y2) & (torch.maximum(y1, y3) > y2)
    add_h(l_case, y1, x1, x2)
    add_h(l_case, y2, x2, x3)
    add_v(l_case, torch.minimum(y1, y2), torch.maximum(y1, y2), x2)
    add_v(l_case, torch.minimum(y2, y3), torch.maximum(y2, y3), x3)

    x23_case = (~l_case) & (x2 == x3) & (x1 < x2) & (y1 < torch.minimum(y2, y3))
    add_h(x23_case, y1, x1, x2)
    add_v(x23_case, y1, torch.maximum(y2, y3), x2)

    y23_case = (~l_case) & (~x23_case) & (y2 == y3)
    add_h(y23_case, y1, x1, x2)
    add_h(y23_case, y2, x2, x3)
    add_v(y23_case, torch.minimum(y2, y1), torch.maximum(y2, y1), x2)

    t_case = ~(l_case | x23_case | y23_case)
    if torch.any(t_case):
        t_idx = torch.nonzero(t_case, as_tuple=False).squeeze(-1)
        y_t = y[t_idx]
        x_t = x[t_idx]
        t_order = torch.argsort(y_t * cols + x_t, dim=-1)
        y_t = torch.gather(y_t, 1, t_order)
        x_t = torch.gather(x_t, 1, t_order)
        ty1, ty2, ty3 = y_t.unbind(dim=-1)
        tx1, tx2, tx3 = x_t.unbind(dim=-1)
        xmin = torch.minimum(torch.minimum(tx1, tx2), tx3)
        xmax = torch.maximum(torch.maximum(tx1, tx2), tx3)
        h_batch.append(batch_idx[t_idx])
        h_row.append(ty2)
        h_start.append(xmin)
        h_end.append(xmax)
        h_weight.append(weights[t_idx])
        v_batch.append(batch_idx[t_idx])
        v_start.append(torch.minimum(ty1, ty2))
        v_end.append(torch.maximum(ty1, ty2))
        v_col.append(tx1)
        v_weight.append(weights[t_idx])
        v_batch.append(batch_idx[t_idx])
        v_start.append(torch.minimum(ty2, ty3))
        v_end.append(torch.maximum(ty2, ty3))
        v_col.append(tx3)
        v_weight.append(weights[t_idx])

    def cat_or_empty(parts, like: torch.Tensor) -> torch.Tensor:
        if parts:
            return torch.cat(parts, dim=0)
        return like.new_zeros((0,), dtype=like.dtype)

    h_map = _scatter_horizontal_segments(
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        batch_idx=cat_or_empty(h_batch, batch_idx),
        row_idx=cat_or_empty(h_row, batch_idx),
        col_start=cat_or_empty(h_start, batch_idx),
        col_end=cat_or_empty(h_end, batch_idx),
        weights=cat_or_empty(h_weight, weights),
    )
    v_map = _scatter_vertical_segments(
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        batch_idx=cat_or_empty(v_batch, batch_idx),
        row_start=cat_or_empty(v_start, batch_idx),
        row_end=cat_or_empty(v_end, batch_idx),
        col_idx=cat_or_empty(v_col, batch_idx),
        weights=cat_or_empty(v_weight, weights),
    )
    return v_map, h_map


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


def _lrouting_directional_maps(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pin_pos = compose_pin_positions(placement, benchmark, netlist)
    pin_pos_b, squeezed = ensure_batched(pin_pos)
    batch_size = pin_pos_b.shape[0]
    rows = benchmark.grid_rows
    cols = benchmark.grid_cols

    pin_indices = netlist.net_pins.clamp_min(0)
    net_pin_pos = pin_pos_b[:, pin_indices]
    mask = netlist.net_mask.unsqueeze(0)
    _, _, net_cell_ids = _pin_pos_to_grid_cells(net_pin_pos, benchmark, netlist)

    pad_cell = rows * cols
    masked_cell_ids = torch.where(mask, net_cell_ids, torch.full_like(net_cell_ids, pad_cell))
    sorted_ids, _ = torch.sort(masked_cell_ids, dim=-1)
    prev_ids = torch.cat(
        [torch.full_like(sorted_ids[..., :1], pad_cell), sorted_ids[..., :-1]],
        dim=-1,
    )
    unique_mask = (sorted_ids != pad_cell) & (sorted_ids != prev_ids)
    unique_count = unique_mask.sum(dim=-1)
    unique_ids = torch.where(unique_mask, sorted_ids, torch.full_like(sorted_ids, pad_cell))
    unique_ids, _ = torch.sort(unique_ids, dim=-1)

    pair_src = pin_pos_b[:, netlist.two_pin_pairs[:, 0]] if netlist.two_pin_pairs.numel() else pin_pos_b.new_zeros((batch_size, 0, 2))
    pair_dst = pin_pos_b[:, netlist.two_pin_pairs[:, 1]] if netlist.two_pin_pairs.numel() else pin_pos_b.new_zeros((batch_size, 0, 2))
    src_row, src_col, _ = _pin_pos_to_grid_cells(pair_src, benchmark, netlist)
    dst_row, dst_col, _ = _pin_pos_to_grid_cells(pair_dst, benchmark, netlist)

    v_two_pin, h_two_pin = _accumulate_two_pin_maps(
        src_row=src_row,
        src_col=src_col,
        dst_row=dst_row,
        dst_col=dst_col,
        net_ids=netlist.two_pin_net_ids,
        unique_count=unique_count,
        weights=netlist.two_pin_weights.to(dtype=pin_pos_b.dtype),
        batch_size=batch_size,
        rows=rows,
        cols=cols,
    )
    v_three_pin, h_three_pin = _accumulate_three_pin_maps(
        unique_ids=unique_ids[..., :3],
        unique_count=unique_count,
        net_weights=netlist.net_weights,
        batch_size=batch_size,
        rows=rows,
        cols=cols,
        dtype=pin_pos_b.dtype,
    )

    v_total = (v_two_pin + v_three_pin) / netlist.grid_v_routes
    h_total = (h_two_pin + h_three_pin) / netlist.grid_h_routes
    if squeezed:
        return v_total.squeeze(0).reshape(-1), h_total.squeeze(0).reshape(-1)
    return v_total.reshape(batch_size, -1), h_total.reshape(batch_size, -1)


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
    cell_rows = (
        torch.arange(benchmark.grid_rows, device=device, dtype=torch.long)
        .repeat_interleave(benchmark.grid_cols)
        .view(1, 1, -1)
    )
    cell_cols = (
        torch.arange(benchmark.grid_cols, device=device, dtype=torch.long)
        .repeat(benchmark.grid_rows)
        .view(1, 1, -1)
    )

    v_map = torch.zeros(
        (batched.shape[0], netlist.grid_xl.numel()),
        dtype=batched.dtype,
        device=device,
    )
    h_map = torch.zeros_like(v_map)

    for start in range(0, benchmark.num_hard_macros, chunk_size):
        end = min(start + chunk_size, benchmark.num_hard_macros)
        xl_chunk = xl[:, start:end].unsqueeze(-1)
        xh_chunk = xh[:, start:end].unsqueeze(-1)
        yl_chunk = yl[:, start:end].unsqueeze(-1)
        yh_chunk = yh[:, start:end].unsqueeze(-1)
        overlap_x = (
            torch.minimum(xh_chunk, cell_xh) - torch.maximum(xl_chunk, cell_xl)
        ).clamp_min(0.0)
        overlap_y = (
            torch.minimum(yh_chunk, cell_yh) - torch.maximum(yl_chunk, cell_yl)
        ).clamp_min(0.0)
        overlap_mask = (overlap_x > 0.0) & (overlap_y > 0.0)
        x_dist = torch.where(overlap_mask, overlap_x, torch.zeros_like(overlap_x))
        y_dist = torch.where(overlap_mask, overlap_y, torch.zeros_like(overlap_y))

        bl_row = torch.floor(yl[:, start:end] / netlist.grid_height).to(torch.long).clamp(
            0, benchmark.grid_rows - 1
        )
        ur_row = torch.floor(yh[:, start:end] / netlist.grid_height).to(torch.long).clamp(
            0, benchmark.grid_rows - 1
        )
        bl_col = torch.floor(xl[:, start:end] / netlist.grid_width).to(torch.long).clamp(
            0, benchmark.grid_cols - 1
        )
        ur_col = torch.floor(xh[:, start:end] / netlist.grid_width).to(torch.long).clamp(
            0, benchmark.grid_cols - 1
        )

        partial_vertical = (
            (ur_row != bl_row)
            & (
                (
                    (cell_rows == bl_row.unsqueeze(-1)) | (cell_rows == ur_row.unsqueeze(-1))
                )
                & (y_dist > 0.0)
                & (y_dist.sub(netlist.grid_height).abs() > 1.0e-5)
            ).any(dim=-1)
        )
        partial_horizontal = (
            (ur_col != bl_col)
            & (
                (
                    (cell_cols == bl_col.unsqueeze(-1)) | (cell_cols == ur_col.unsqueeze(-1))
                )
                & (x_dist > 0.0)
                & (x_dist.sub(netlist.grid_width).abs() > 1.0e-5)
            ).any(dim=-1)
        )

        v_map = v_map + x_dist.sum(dim=1)
        h_map = h_map + y_dist.sum(dim=1)
        v_map = v_map - (
            x_dist
            * partial_vertical.unsqueeze(-1)
            * (cell_rows == ur_row.unsqueeze(-1))
        ).sum(dim=1)
        h_map = h_map - (
            y_dist
            * partial_horizontal.unsqueeze(-1)
            * (cell_cols == ur_col.unsqueeze(-1))
        ).sum(dim=1)

    v_map = v_map * (netlist.vrouting_alloc / max(netlist.grid_v_routes, 1.0e-6))
    h_map = h_map * (netlist.hrouting_alloc / max(netlist.grid_h_routes, 1.0e-6))
    return unbatch(v_map, squeezed), unbatch(h_map, squeezed)


def _smooth_directional_map(direction_map: torch.Tensor, axis: int, radius: int) -> torch.Tensor:
    if radius <= 0:
        return direction_map
    batched, squeezed = ensure_grid_batched(direction_map)
    if axis == 1:
        length = batched.shape[-1]
        view = batched.reshape(-1, 1, length)
    else:
        length = batched.shape[-2]
        view = batched.transpose(1, 2).reshape(-1, 1, length)

    idx = torch.arange(length, device=batched.device)
    span = (
        torch.minimum(idx + radius, torch.full_like(idx, length - 1))
        - torch.maximum(idx - radius, torch.zeros_like(idx))
        + 1
    ).to(dtype=batched.dtype)
    kernel = torch.ones((1, 1, 2 * radius + 1), dtype=batched.dtype, device=batched.device)
    out = F.conv1d(view / span.view(1, 1, -1), kernel, padding=radius)
    if axis == 1:
        out = out.reshape_as(batched)
    else:
        out = out.reshape(batched.shape[0], batched.shape[2], batched.shape[1]).transpose(1, 2)
    return unbatch(out, squeezed)


def compute_congestion_maps(
    placement: torch.Tensor,
    benchmark: Benchmark,
    netlist: NetlistTensors,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_net, h_net = _lrouting_directional_maps(placement, benchmark, netlist)
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
    cost = values.mean(dim=-1)
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
