from __future__ import annotations

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _clamp_point(
    x: float,
    y: float,
    half_w: float,
    half_h: float,
    canvas_w: float,
    canvas_h: float,
) -> np.ndarray:
    return np.array(
        [
            np.clip(x, half_w, canvas_w - half_w),
            np.clip(y, half_h, canvas_h - half_h),
        ],
        dtype=np.float64,
    )


def _macro_overlap_areas(
    idx: int,
    candidate: np.ndarray,
    positions: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
) -> np.ndarray:
    dx = np.abs(candidate[0] - positions[:, 0])
    dy = np.abs(candidate[1] - positions[:, 1])
    overlap_x = np.maximum(0.0, sep_x[idx] - dx)
    overlap_y = np.maximum(0.0, sep_y[idx] - dy)
    overlap = overlap_x * overlap_y
    overlap[idx] = 0.0
    return overlap


def _pairwise_overlap_matrix(positions: np.ndarray, sep_x: np.ndarray, sep_y: np.ndarray) -> np.ndarray:
    dx = np.abs(positions[:, 0:1] - positions[:, 0:1].T)
    dy = np.abs(positions[:, 1:2] - positions[:, 1:2].T)
    overlap_x = np.maximum(0.0, sep_x - dx)
    overlap_y = np.maximum(0.0, sep_y - dy)
    overlap = overlap_x * overlap_y
    np.fill_diagonal(overlap, 0.0)
    return overlap


def _quadrant_index(candidate: np.ndarray, canvas_w: float, canvas_h: float) -> int:
    right = int(candidate[0] >= canvas_w * 0.5)
    top = int(candidate[1] >= canvas_h * 0.5)
    return top * 2 + right


def _search_near_center(
    idx: int,
    center: np.ndarray,
    anchor: np.ndarray,
    current: np.ndarray,
    positions: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    step: float,
    radius: int,
) -> tuple[tuple[float, float, float, float] | None, np.ndarray | None, tuple[float, float, float, float], np.ndarray]:
    best_legal_key = None
    best_legal_pos = None
    best_partial_key = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_partial_pos = current.copy()
    quadrant_counts = np.zeros(4, dtype=np.float64)
    for other_idx, other in enumerate(positions):
        if other_idx == idx:
            continue
        quadrant_counts[_quadrant_index(other, canvas_w, canvas_h)] += 1.0

    for dxm in range(-radius, radius + 1):
        for dym in range(-radius, radius + 1):
            candidate = _clamp_point(
                center[0] + dxm * step,
                center[1] + dym * step,
                half_w[idx],
                half_h[idx],
                canvas_w,
                canvas_h,
            )
            overlap = _macro_overlap_areas(idx, candidate, positions, sep_x, sep_y)
            total_overlap = float(overlap.sum())
            overlap_count = float(np.count_nonzero(overlap > 0.0))
            disp_anchor = float(np.sum((candidate - anchor) ** 2))
            disp_current = float(np.sum((candidate - current) ** 2))
            density_penalty = float(quadrant_counts[_quadrant_index(candidate, canvas_w, canvas_h)])
            key = (disp_anchor + density_penalty * 0.1, disp_current, overlap_count, total_overlap)
            if total_overlap == 0.0:
                if best_legal_key is None or key < best_legal_key:
                    best_legal_key = key
                    best_legal_pos = candidate
                continue
            partial_key = (
                total_overlap,
                overlap_count,
                disp_anchor + density_penalty * 0.1,
                disp_current,
            )
            if partial_key < best_partial_key:
                best_partial_key = partial_key
                best_partial_pos = candidate

    return best_legal_key, best_legal_pos, best_partial_key, best_partial_pos


def _repair_macro(
    idx: int,
    anchors: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    max_radius: int,
) -> tuple[np.ndarray, bool]:
    anchor = anchors[idx]
    current = positions[idx].copy()
    base_step = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.2
    base_step = max(base_step, 1.0)
    search_plan = [
        (anchor, base_step, min(max_radius, 8)),
        (anchor, base_step, min(max_radius, 16)),
        (current, base_step, min(max_radius, 16)),
        (anchor, base_step * 0.5, min(max_radius, 32)),
        (current, base_step * 0.5, min(max_radius, 32)),
    ]

    best_legal_key = None
    best_legal_pos = None
    best_partial_key = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_partial_pos = current.copy()

    for center, step, radius in search_plan:
        legal_key, legal_pos, partial_key, partial_pos = _search_near_center(
            idx,
            center,
            anchor,
            current,
            positions,
            sep_x,
            sep_y,
            half_w,
            half_h,
            canvas_w,
            canvas_h,
            step,
            radius,
        )
        if legal_key is not None and (best_legal_key is None or legal_key < best_legal_key):
            best_legal_key = legal_key
            best_legal_pos = legal_pos
        if partial_key < best_partial_key:
            best_partial_key = partial_key
            best_partial_pos = partial_pos

    if best_legal_pos is None:
        return best_partial_pos, False

    fine_center = best_legal_pos.copy()
    fine_step = max(base_step * 0.125, 0.25)
    fine_radius = min(max_radius, 6)
    fine_legal_key, fine_legal_pos, _, _ = _search_near_center(
        idx,
        fine_center,
        anchor,
        current,
        positions,
        sep_x,
        sep_y,
        half_w,
        half_h,
        canvas_w,
        canvas_h,
        fine_step,
        fine_radius,
    )
    if fine_legal_key is not None and fine_legal_key < best_legal_key:
        best_legal_pos = fine_legal_pos

    return best_legal_pos, True


def _legacy_ring_legalize(
    out: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    max_radius: int,
) -> np.ndarray:
    repaired = out.copy()
    n_hard = repaired.shape[0]
    placed = np.zeros(n_hard, dtype=bool)
    order = sorted(range(n_hard), key=lambda idx: -(sizes[idx, 0] * sizes[idx, 1]))

    for idx in order:
        if not movable[idx]:
            placed[idx] = True
            continue

        def is_legal(candidate_x: float, candidate_y: float) -> bool:
            if not placed.any():
                return True
            dx = np.abs(candidate_x - repaired[:, 0])
            dy = np.abs(candidate_y - repaired[:, 1])
            overlaps = (dx < sep_x[idx]) & (dy < sep_y[idx]) & placed
            overlaps[idx] = False
            return not overlaps.any()

        cur = _clamp_point(
            repaired[idx, 0],
            repaired[idx, 1],
            half_w[idx],
            half_h[idx],
            canvas_w,
            canvas_h,
        )
        if is_legal(cur[0], cur[1]):
            repaired[idx] = cur
            placed[idx] = True
            continue

        step = max(float(sizes[idx, 0]), float(sizes[idx, 1])) * 0.25
        step = max(step, 1.0)
        best = cur.copy()
        best_dist = float("inf")
        for radius in range(1, max_radius + 1):
            found = False
            for dxm in range(-radius, radius + 1):
                for dym in range(-radius, radius + 1):
                    if abs(dxm) != radius and abs(dym) != radius:
                        continue
                    cand = _clamp_point(
                        cur[0] + dxm * step,
                        cur[1] + dym * step,
                        half_w[idx],
                        half_h[idx],
                        canvas_w,
                        canvas_h,
                    )
                    if not is_legal(cand[0], cand[1]):
                        continue
                    dist = float(np.sum((cand - cur) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best = cand
                        found = True
            if found:
                break
        repaired[idx] = best
        placed[idx] = True

    return repaired


def _anchored_legalize(
    anchors: np.ndarray,
    movable: np.ndarray,
    sizes: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    half_w: np.ndarray,
    half_h: np.ndarray,
    canvas_w: float,
    canvas_h: float,
    max_radius: int,
) -> tuple[np.ndarray, int]:
    repaired = anchors.copy()
    max_passes = max(8, min(64, repaired.shape[0] * 2))
    passes = 0
    for passes in range(1, max_passes + 1):
        overlap = _pairwise_overlap_matrix(repaired, sep_x, sep_y)
        per_macro_overlap = overlap.sum(axis=1)
        overlapping = np.where(per_macro_overlap > 0.0)[0]
        if overlapping.size == 0:
            break

        movable_overlapping = [idx for idx in overlapping.tolist() if movable[idx]]
        if not movable_overlapping:
            break

        order = sorted(
            movable_overlapping,
            key=lambda idx: (-per_macro_overlap[idx], -(sizes[idx, 0] * sizes[idx, 1]), idx),
        )
        progressed = False
        for idx in order:
            current_overlap = _macro_overlap_areas(idx, repaired[idx], repaired, sep_x, sep_y).sum()
            if current_overlap == 0.0:
                continue
            candidate, legal = _repair_macro(
                idx,
                anchors,
                repaired,
                sizes,
                sep_x,
                sep_y,
                half_w,
                half_h,
                canvas_w,
                canvas_h,
                max_radius,
            )
            if legal or not np.allclose(candidate, repaired[idx]):
                new_overlap = _macro_overlap_areas(idx, candidate, repaired, sep_x, sep_y).sum()
                if new_overlap < current_overlap or legal:
                    repaired[idx] = candidate
                    progressed = True
        if not progressed:
            break
    return repaired, passes


def _collect_legalize_stats(
    repaired: np.ndarray,
    anchors: np.ndarray,
    movable: np.ndarray,
    sep_x: np.ndarray,
    sep_y: np.ndarray,
    passes: int,
    method: str,
) -> dict:
    final_overlap = _pairwise_overlap_matrix(repaired, sep_x, sep_y)
    hard_delta = np.linalg.norm(repaired - anchors, axis=1)
    moved_mask = hard_delta > 1.0e-6
    upper_overlap = np.triu(final_overlap, k=1)
    return {
        "method": method,
        "total_hard_displacement": float(hard_delta[movable].sum()),
        "max_hard_displacement": float(hard_delta[movable].max()) if np.any(movable) else 0.0,
        "moved_hard_macros": int(np.count_nonzero(moved_mask & movable)),
        "repair_passes": passes,
        "remaining_overlap_count": int(np.count_nonzero(upper_overlap > 0.0)),
        "remaining_overlap_area": float(upper_overlap.sum()),
    }


def _select_legalized_result(candidates: list[tuple[np.ndarray, dict]]) -> tuple[np.ndarray, dict]:
    def sort_key(item: tuple[np.ndarray, dict]) -> tuple[float, float, float]:
        stats = item[1]
        return (
            float(stats["remaining_overlap_area"]),
            float(stats["total_hard_displacement"]),
            float(stats["max_hard_displacement"]),
        )

    return min(candidates, key=sort_key)


def legalize_hard_macro_variants(
    placement: torch.Tensor,
    benchmark: Benchmark,
    safety_gap: float = 0.005,
    max_radius: int = 200,
) -> list[dict]:
    out = placement.detach().cpu().numpy().copy().astype(np.float64)
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable = benchmark.get_movable_mask()[:n_hard].cpu().numpy()
    half_w = sizes[:, 0] / 2.0
    half_h = sizes[:, 1] / 2.0
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2.0 + safety_gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2.0 + safety_gap
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    anchors = out[:n_hard].copy()
    for idx in range(n_hard):
        anchors[idx] = _clamp_point(anchors[idx, 0], anchors[idx, 1], half_w[idx], half_h[idx], canvas_w, canvas_h)

    anchored_out, anchored_passes = _anchored_legalize(
        anchors,
        movable,
        sizes,
        sep_x,
        sep_y,
        half_w,
        half_h,
        canvas_w,
        canvas_h,
        max_radius,
    )
    anchored_stats = _collect_legalize_stats(
        anchored_out,
        anchors,
        movable,
        sep_x,
        sep_y,
        anchored_passes,
        method="anchored",
    )

    legacy_out = _legacy_ring_legalize(
        anchors,
        movable,
        sizes,
        half_w,
        half_h,
        sep_x,
        sep_y,
        canvas_w,
        canvas_h,
        max_radius=max_radius,
    )
    legacy_stats = _collect_legalize_stats(
        legacy_out,
        anchors,
        movable,
        sep_x,
        sep_y,
        passes=1,
        method="legacy",
    )

    variants = []
    for repaired, stats in [
        (anchored_out, anchored_stats),
        (legacy_out, legacy_stats),
    ]:
        legalized = placement.clone()
        legalized[:n_hard] = torch.tensor(repaired, dtype=placement.dtype)
        variants.append(
            {
                "method": stats["method"],
                "placement": legalized,
                "stats": stats,
            }
        )
    return variants


def legalize_hard_macros(
    placement: torch.Tensor,
    benchmark: Benchmark,
    safety_gap: float = 0.005,
    max_radius: int = 200,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    variants = legalize_hard_macro_variants(
        placement,
        benchmark,
        safety_gap=safety_gap,
        max_radius=max_radius,
    )
    legalized, stats = _select_legalized_result(
        [(variant["placement"][: benchmark.num_hard_macros].detach().cpu().numpy(), variant["stats"]) for variant in variants]
    )
    out = placement.clone()
    out[: benchmark.num_hard_macros] = torch.tensor(legalized, dtype=placement.dtype)
    stats = {
        **stats,
        "selected_method": stats["method"],
        "candidate_methods": [variant["method"] for variant in variants],
    }
    if return_stats:
        return out, stats
    return out
