from __future__ import annotations

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
from fd_soft import optimize_soft_macros
from gpu_cost import compute_proxy_cost, get_default_device
from legalize import legalize_hard_macros
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
    ):
        self.seed = seed
        self.analytical_starts = analytical_starts
        self.analytical_iters = analytical_iters
        self.sa_chains = sa_chains
        self.sa_steps = sa_steps
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

        candidates = []
        initial = benchmark.macro_positions.clone()
        candidates.append(("initial", initial))
        candidates.append(("initial_legalized", legalize_hard_macros(initial, benchmark)))

        candidate_iters = self._candidate_iters(benchmark)
        for iters in candidate_iters:
            analytical = run_analytical_placement(
                benchmark,
                netlist,
                self.device,
                num_starts=self._candidate_starts(benchmark),
                num_iters=iters,
                seed=self.seed,
            )
            candidates.append((f"analytical_{iters}", analytical["best_placement"]))

        best_name = None
        best_score = float("inf")
        best_placement = initial
        for name, candidate in candidates:
            exact = exact_proxy_cost(candidate, benchmark, plc)
            if int(exact["overlap_count"]) != 0:
                continue
            if float(exact["proxy_cost"]) < best_score:
                best_name = name
                best_score = float(exact["proxy_cost"])
                best_placement = candidate.detach().clone()

        if best_name is None:
            best_name = "initial_legalized"
            best_placement = legalize_hard_macros(initial, benchmark)
            best_score = float(exact_proxy_cost(best_placement, benchmark, plc)["proxy_cost"])

        selected_placement = best_placement.detach().clone()

        if self._use_soft_refinement(benchmark):
            soft_refined = optimize_soft_macros(
                selected_placement.to(self.device),
                benchmark,
                netlist,
                num_steps=20 if benchmark.num_hard_macros > 350 else 30,
            ).cpu()
            soft_exact = exact_proxy_cost(soft_refined, benchmark, plc)
            if int(soft_exact["overlap_count"]) == 0 and float(soft_exact["proxy_cost"]) < best_score:
                best_name = "soft_refined"
                best_score = float(soft_exact["proxy_cost"])
                selected_placement = soft_refined.detach().clone()

        if self._use_sa_refinement(benchmark):
            sa_candidate = run_parallel_sa(
                selected_placement,
                benchmark,
                netlist,
                self.device,
                num_chains=self._sa_chains(benchmark),
                max_steps=self._sa_steps(benchmark),
                seed=self.seed,
            )
            sa_exact = exact_proxy_cost(sa_candidate, benchmark, plc)
            if int(sa_exact["overlap_count"]) == 0 and float(sa_exact["proxy_cost"]) < best_score:
                best_name = "sa_refined"
                best_score = float(sa_exact["proxy_cost"])
                selected_placement = sa_candidate.detach().clone()

        placement = selected_placement.detach().clone()
        gpu_cost = compute_proxy_cost(placement.to(self.device), benchmark, netlist)
        print(
            f"[GPUPlacer] {benchmark.name}: "
            f"picked={best_name} "
            f"exact_proxy={best_score:.4f} "
            f"proxy={float(gpu_cost['proxy_cost']):.4f} "
            f"wl={float(gpu_cost['wirelength_cost']):.4f} "
            f"den={float(gpu_cost['density_cost']):.4f} "
            f"cong={float(gpu_cost['congestion_cost']):.4f} "
            f"device={self.device}"
        )
        return placement

    def _candidate_iters(self, benchmark: Benchmark):
        if benchmark.num_hard_macros >= 450:
            return [5]
        if benchmark.num_hard_macros >= 320:
            return [5, 10]
        return sorted({5, 10, self.analytical_iters})

    def _candidate_starts(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 400:
            return min(self.analytical_starts, 2)
        if benchmark.num_hard_macros >= 300:
            return min(self.analytical_starts, 3)
        return self.analytical_starts

    def _use_soft_refinement(self, benchmark: Benchmark) -> bool:
        return benchmark.num_hard_macros < 400

    def _use_sa_refinement(self, benchmark: Benchmark) -> bool:
        return benchmark.num_hard_macros < 300

    def _sa_chains(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 200:
            return min(self.sa_chains, 6)
        return self.sa_chains

    def _sa_steps(self, benchmark: Benchmark) -> int:
        if benchmark.num_hard_macros >= 200:
            return min(self.sa_steps, 60)
        return self.sa_steps
