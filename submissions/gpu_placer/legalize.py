from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def legalize_hard_macros(
    placement: torch.Tensor,
    benchmark: Benchmark,
    safety_gap: float = 0.05,
    max_radius: int = 200,
) -> torch.Tensor:
    out = placement.detach().cpu().numpy().copy().astype(np.float64)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0
    placed = np.zeros(n_hard, dtype=bool)
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    order = sorted(range(n_hard), key=lambda idx: -(sizes[idx, 0] * sizes[idx, 1]))
    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        def is_legal(candidate_x: float, candidate_y: float) -> bool:
            if placed.any():
                dx = np.abs(candidate_x - out[:n_hard, 0])
                dy = np.abs(candidate_y - out[:n_hard, 1])
                overlaps = (
                    (dx < sep_x[idx] + safety_gap)
                    & (dy < sep_y[idx] + safety_gap)
                    & placed
                )
                overlaps[idx] = False
                return not overlaps.any()
            return True

        cur_x = np.clip(out[idx, 0], half_w[idx], canvas_w - half_w[idx])
        cur_y = np.clip(out[idx, 1], half_h[idx], canvas_h - half_h[idx])
        if is_legal(cur_x, cur_y):
            out[idx, 0] = cur_x
            out[idx, 1] = cur_y
            placed[idx] = True
            continue

        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
        best = np.array([cur_x, cur_y])
        best_dist = float("inf")
        for radius in range(1, max_radius + 1):
            found = False
            for dxm in range(-radius, radius + 1):
                for dym in range(-radius, radius + 1):
                    if abs(dxm) != radius and abs(dym) != radius:
                        continue
                    cand_x = np.clip(cur_x + dxm * step, half_w[idx], canvas_w - half_w[idx])
                    cand_y = np.clip(cur_y + dym * step, half_h[idx], canvas_h - half_h[idx])
                    if not is_legal(cand_x, cand_y):
                        continue
                    dist = (cand_x - cur_x) ** 2 + (cand_y - cur_y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best = np.array([cand_x, cand_y])
                        found = True
            if found:
                break

        out[idx, 0] = best[0]
        out[idx, 1] = best[1]
        placed[idx] = True

    legalized = placement.clone()
    legalized[:n_hard] = torch.tensor(out[:n_hard], dtype=placement.dtype)
    return legalized
