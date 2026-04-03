from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost as exact_proxy_cost

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analytical import run_analytical_placement
from gpu_cost import compute_proxy_cost, get_default_device
from legalize import legalize_hard_macro_variants
from net_extract import build_netlist_tensors
from sa_refine import run_parallel_sa


def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir

    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc

    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
        "ariane133": "ariane133",
        "ariane136": "ariane136",
        "nvdla": "nvdla",
        "mempool_tile": "mempool_tile",
    }
    mapped = ng45.get(name)
    if mapped:
        base = (
            Path("external/MacroPlacement/Flows/NanGate45")
            / mapped
            / "netlist"
            / "output_CT_Grouping"
        )
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None


def _resolve_benchmark_key(benchmark: Benchmark) -> str:
    if benchmark.name != "output_CT_Grouping":
        return benchmark.name

    signature_map = {
        (133, 22584): "ariane133",
        (136, 23067): "ariane136",
        (128, 40606): "nvdla",
        (20, 32944): "mempool_tile",
    }
    return signature_map.get((benchmark.num_hard_macros, benchmark.num_nets), benchmark.name)


class GPUPlacer:
    def __init__(
        self,
        seed: int = 42,
        analytical_starts: int = 4,
        analytical_iters: int = 20,
        sa_chains: int = 8,
        sa_steps: int = 100,
        candidate_debug: bool | None = None,
    ):
        self.seed = seed
        self.analytical_starts = analytical_starts
        self.analytical_iters = analytical_iters
        self.sa_chains = sa_chains
        self.sa_steps = sa_steps
        self.candidate_debug = (
            os.environ.get("GPU_PLACER_DEBUG_CANDIDATES", "0") == "1"
            if candidate_debug is None
            else candidate_debug
        )
        self.device = get_default_device()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        benchmark_key = _resolve_benchmark_key(benchmark)
        plc = _load_plc(benchmark_key)
        if plc is None:
            return benchmark.macro_positions.clone()

        netlist = build_netlist_tensors(benchmark, plc, device=self.device)

        candidate_records = []
        initial = benchmark.macro_positions.clone()
        candidate_records.append(
            self._score_candidate(
                "initial",
                initial,
                benchmark,
                plc,
                stage="seed",
                notes="raw benchmark placement",
            )
        )
        initial_legalized_records = self._score_legalize_variants(
            "initial_legalized",
            legalize_hard_macro_variants(initial, benchmark),
            benchmark,
            plc,
            stage="seed",
            label="legalize",
        )
        candidate_records.extend(initial_legalized_records)
        initial_legalized_record = self._select_best_candidate(initial_legalized_records)

        candidate_iters = self._analytical_iters(benchmark)
        analytical_starts = self._analytical_starts(benchmark)
        for iters in candidate_iters:
            analytical = run_analytical_placement(
                benchmark,
                netlist,
                self.device,
                num_starts=analytical_starts,
                num_iters=iters,
                seed=self.seed,
                time_budget_s=self._analytical_time_budget_s(benchmark),
            )
            candidate_records.extend(
                self._score_legalize_variants(
                    f"analytical_{iters}",
                    analytical["legalization_variants"],
                    benchmark,
                    plc,
                    stage="analytical",
                    prefix=(
                        f"starts={analytical_starts} steps={analytical['executed_steps']}"
                    ),
                    label="legalize",
                )
            )

        best_record = self._select_best_candidate(candidate_records)
        selected_placement = best_record["placement"].detach().clone()

        if self._should_run_post_legal_refinement(benchmark, best_record):
            refine_config = self._post_legal_refinement_config(benchmark)
            post_legal = run_analytical_placement(
                benchmark,
                netlist,
                self.device,
                num_starts=refine_config["starts"],
                num_iters=refine_config["iters"],
                seed=self.seed + 7,
                time_budget_s=refine_config["time_budget_s"],
                seed_positions=[selected_placement],
                overlap_weight_start=20.0,
                overlap_weight_end=120.0,
                density_weight_start=0.45,
                density_weight_end=0.30,
                congestion_weight_start=1.0,
                congestion_weight_end=1.1,
            )
            candidate_records.extend(
                self._score_legalize_variants(
                    f"{best_record['name']}_post_legal",
                    post_legal["legalization_variants"],
                    benchmark,
                    plc,
                    stage="post_legal_refine",
                    prefix=(
                        f"starts={refine_config['starts']} steps={post_legal['executed_steps']}"
                    ),
                    label="legalize",
                )
            )
            best_record = self._select_best_candidate(candidate_records)
            selected_placement = best_record["placement"].detach().clone()

        if self._should_run_sa_refinement(benchmark, best_record, initial_legalized_record):
            sa_candidate = run_parallel_sa(
                selected_placement,
                benchmark,
                netlist,
                self.device,
                num_chains=self._sa_chains(benchmark),
                max_steps=self._sa_steps(benchmark),
                time_budget_s=self._sa_time_budget_s(benchmark),
                seed=self.seed,
            )
            sa_record = self._score_candidate(
                "sa_refined",
                sa_candidate,
                benchmark,
                plc,
                stage="sa_refine",
                notes=f"chains={self._sa_chains(benchmark)} steps={self._sa_steps(benchmark)}",
            )
            candidate_records.append(sa_record)
            if self._is_better_record(sa_record, best_record):
                best_record = sa_record
                selected_placement = sa_candidate.detach().clone()

        placement = selected_placement.detach().clone()
        gpu_cost = compute_proxy_cost(placement.to(self.device), benchmark, netlist)
        self._log_candidate_diagnostics(benchmark, candidate_records, best_record["name"])
        print(
            f"[GPUPlacer] {benchmark.name}: "
            f"picked={best_record['name']} "
            f"exact_proxy={best_record['proxy_cost']:.4f} "
            f"proxy={float(gpu_cost['proxy_cost']):.4f} "
            f"wl={float(gpu_cost['wirelength_cost']):.4f} "
            f"den={float(gpu_cost['density_cost']):.4f} "
            f"cong={float(gpu_cost['congestion_cost']):.4f} "
            f"device={self.device}"
        )
        return placement

    def _score_candidate(
        self,
        name: str,
        candidate: torch.Tensor,
        benchmark: Benchmark,
        plc,
        *,
        stage: str,
        notes: str = "",
    ) -> dict:
        placement = candidate.detach().clone()
        exact = exact_proxy_cost(placement, benchmark, plc)
        return {
            "name": name,
            "placement": placement,
            "stage": stage,
            "notes": notes,
            "proxy_cost": float(exact["proxy_cost"]),
            "wirelength_cost": float(exact["wirelength_cost"]),
            "density_cost": float(exact["density_cost"]),
            "congestion_cost": float(exact["congestion_cost"]),
            "overlap_count": int(exact["overlap_count"]),
        }

    def _score_legalize_variants(
        self,
        base_name: str,
        variants: list[dict],
        benchmark: Benchmark,
        plc,
        *,
        stage: str,
        label: str,
        prefix: str = "",
    ) -> list[dict]:
        records = []
        for variant in variants:
            note = self._format_legalize_stats(f"{label} {variant['method']}", variant["stats"])
            if prefix:
                note = f"{prefix} {note}"
            record = self._score_candidate(
                    f"{base_name}_{variant['method']}",
                    variant["placement"],
                    benchmark,
                    plc,
                    stage=stage,
                    notes=note,
            )
            record["legalize_stats"] = variant["stats"]
            records.append(record)
        return records

    def _select_best_candidate(self, records: list[dict]) -> dict:
        best_record = None
        best_score = float("inf")
        for record in records:
            if record["overlap_count"] != 0:
                continue
            if record["proxy_cost"] < best_score:
                best_score = record["proxy_cost"]
                best_record = record
        if best_record is None:
            best_record = min(
                records,
                key=lambda record: (record["overlap_count"], record["proxy_cost"]),
            )
        return best_record

    def _is_better_record(self, candidate: dict, incumbent: dict) -> bool:
        return candidate["overlap_count"] == 0 and candidate["proxy_cost"] < incumbent["proxy_cost"]

    def _should_run_round2(self, benchmark: Benchmark, best_record: dict, baseline_record: dict) -> bool:
        return False

    def _should_run_soft_refinement(
        self,
        benchmark: Benchmark,
        best_record: dict,
        baseline_record: dict,
    ) -> bool:
        return False

    def _should_run_post_legal_refinement(self, benchmark: Benchmark, best_record: dict) -> bool:
        stats = best_record.get("legalize_stats")
        if stats is None or best_record["overlap_count"] != 0:
            return False
        if benchmark.num_hard_macros >= 450:
            min_disp = 220.0
        elif benchmark.num_hard_macros >= 280:
            min_disp = 160.0
        else:
            min_disp = 100.0
        return (
            float(stats["total_hard_displacement"]) >= min_disp
            or float(stats["max_hard_displacement"]) >= 12.0
        )

    def _post_legal_refinement_config(self, benchmark: Benchmark) -> dict:
        if benchmark.num_hard_macros >= 450:
            return {"starts": 2, "iters": 12, "time_budget_s": 8.0}
        if benchmark.num_hard_macros >= 280:
            return {"starts": 2, "iters": 16, "time_budget_s": 10.0}
        return {"starts": 3, "iters": 18, "time_budget_s": 12.0}

    def _should_run_sa_refinement(
        self,
        benchmark: Benchmark,
        best_record: dict,
        baseline_record: dict,
    ) -> bool:
        if best_record["name"].startswith("initial"):
            return False
        if not self._use_sa_refinement(benchmark):
            return False
        improvement = baseline_record["proxy_cost"] - best_record["proxy_cost"]
        return improvement >= 0.008 and best_record["congestion_cost"] >= 1.6

    def _log_candidate_diagnostics(
        self,
        benchmark: Benchmark,
        records: list[dict],
        winner_name: str,
    ) -> None:
        if not self.candidate_debug:
            return
        print(f"[GPUPlacer:candidates] {benchmark.name}")
        for record in records:
            marker = "*" if record["name"] == winner_name else " "
            print(
                f"  {marker} {record['name']:<28} "
                f"stage={record['stage']:<17} "
                f"proxy={record['proxy_cost']:.4f} "
                f"wl={record['wirelength_cost']:.4f} "
                f"den={record['density_cost']:.4f} "
                f"cong={record['congestion_cost']:.4f} "
                f"ov={record['overlap_count']}"
            )
            if record["notes"]:
                print(f"    note={record['notes']}")

    def _format_legalize_stats(self, label: str, stats: dict) -> str:
        return (
            f"{label} moved={stats['moved_hard_macros']} "
            f"total_disp={stats['total_hard_displacement']:.2f} "
            f"max_disp={stats['max_hard_displacement']:.2f} "
            f"passes={stats['repair_passes']} "
            f"rem_ov={stats['remaining_overlap_count']}"
        )

    def _analytical_iters(self, benchmark: Benchmark) -> list[int]:
        if benchmark.num_hard_macros >= 450:
            return [30]
        if benchmark.num_hard_macros >= 320:
            return [50]
        return [max(self.analytical_iters, 80)]

    def _analytical_starts(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 400:
            return 3
        if benchmark.num_hard_macros >= 300:
            return 4
        if benchmark.num_hard_macros >= 200:
            return max(self.analytical_starts, 4)
        return max(self.analytical_starts, 5)

    def _analytical_time_budget_s(self, benchmark: Benchmark) -> float:
        if benchmark.num_hard_macros >= 450:
            return 40.0
        if benchmark.num_hard_macros >= 320:
            return 30.0
        return 25.0

    def _use_soft_refinement(self, benchmark: Benchmark) -> bool:
        return False

    def _use_sa_refinement(self, benchmark: Benchmark) -> bool:
        return benchmark.num_hard_macros < 340

    def _sa_chains(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 280:
            return min(self.sa_chains, 4)
        if benchmark.num_hard_macros >= 200:
            return min(self.sa_chains, 6)
        return self.sa_chains

    def _sa_steps(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 280:
            return min(self.sa_steps, 30)
        if benchmark.num_hard_macros >= 200:
            return min(self.sa_steps, 45)
        return min(self.sa_steps, 50)

    def _sa_time_budget_s(self, benchmark: Benchmark) -> float:
        if benchmark.num_hard_macros >= 280:
            return 12.0
        return 18.0
