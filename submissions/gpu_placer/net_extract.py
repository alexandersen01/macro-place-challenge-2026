from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from macro_place.benchmark import Benchmark


@dataclass
class NetlistTensors:
    pin_parent_idx: torch.Tensor
    pin_offsets: torch.Tensor
    net_pins: torch.Tensor
    net_mask: torch.Tensor
    net_weights: torch.Tensor
    net_cnt: float
    macro_edges: torch.Tensor
    macro_edge_weights: torch.Tensor
    macro_adjacency: List[List[int]]
    macro_adjacency_weights: List[List[float]]
    grid_xl: torch.Tensor
    grid_xh: torch.Tensor
    grid_yl: torch.Tensor
    grid_yh: torch.Tensor
    grid_width: float
    grid_height: float
    grid_v_routes: float
    grid_h_routes: float
    vrouting_alloc: float
    hrouting_alloc: float
    smooth_range: int
    congestion_scale: float = 1.0

    def to(self, device: torch.device) -> "NetlistTensors":
        return NetlistTensors(
            pin_parent_idx=self.pin_parent_idx.to(device),
            pin_offsets=self.pin_offsets.to(device),
            net_pins=self.net_pins.to(device),
            net_mask=self.net_mask.to(device),
            net_weights=self.net_weights.to(device),
            net_cnt=self.net_cnt,
            macro_edges=self.macro_edges.to(device),
            macro_edge_weights=self.macro_edge_weights.to(device),
            macro_adjacency=self.macro_adjacency,
            macro_adjacency_weights=self.macro_adjacency_weights,
            grid_xl=self.grid_xl.to(device),
            grid_xh=self.grid_xh.to(device),
            grid_yl=self.grid_yl.to(device),
            grid_yh=self.grid_yh.to(device),
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            grid_v_routes=self.grid_v_routes,
            grid_h_routes=self.grid_h_routes,
            vrouting_alloc=self.vrouting_alloc,
            hrouting_alloc=self.hrouting_alloc,
            smooth_range=self.smooth_range,
            congestion_scale=self.congestion_scale,
        )


def build_netlist_tensors(
    benchmark: Benchmark,
    plc,
    device: torch.device | str = "cpu",
) -> NetlistTensors:
    device = torch.device(device)

    macro_name_to_idx = {name: idx for idx, name in enumerate(benchmark.macro_names)}
    port_plc_to_offset = {plc_idx: offset for offset, plc_idx in enumerate(plc.port_indices)}

    pin_names: List[str] = []
    pin_parent_idx: List[int] = []
    pin_offsets: List[List[float]] = []
    pin_name_to_pin_idx: Dict[str, int] = {}

    def ensure_pin(pin_name: str) -> int:
        existing = pin_name_to_pin_idx.get(pin_name)
        if existing is not None:
            return existing

        plc_idx = plc.mod_name_to_indices[pin_name]
        node = plc.modules_w_pins[plc_idx]
        node_type = node.get_type()

        if node_type == "PORT":
            parent_idx = benchmark.num_macros + port_plc_to_offset[plc_idx]
            offset = [0.0, 0.0]
        elif node_type in {"MACRO_PIN", "macro_pin"}:
            macro_name = node.get_macro_name()
            parent_idx = macro_name_to_idx[macro_name]
            x_offset = float(getattr(node, "x_offset", 0.0))
            y_offset = float(getattr(node, "y_offset", 0.0))
            offset = [x_offset, y_offset]
        else:
            raise ValueError(f"Unsupported pin node type for {pin_name}: {node_type}")

        pin_idx = len(pin_names)
        pin_names.append(pin_name)
        pin_parent_idx.append(parent_idx)
        pin_offsets.append(offset)
        pin_name_to_pin_idx[pin_name] = pin_idx
        return pin_idx

    net_pin_lists: List[List[int]] = []
    net_weights: List[float] = []

    edge_dict: Dict[Tuple[int, int], float] = {}
    adjacency_pairs: Dict[int, Dict[int, float]] = {
        idx: {} for idx in range(benchmark.num_macros)
    }

    for driver_name, sinks in plc.nets.items():
        driver_idx = ensure_pin(driver_name)
        sink_indices = [ensure_pin(sink_name) for sink_name in sinks]
        pin_list = [driver_idx] + sink_indices
        net_pin_lists.append(pin_list)

        driver_plc_idx = plc.mod_name_to_indices[driver_name]
        driver_node = plc.modules_w_pins[driver_plc_idx]
        net_weights.append(float(driver_node.get_weight()))

        parent_macros = sorted(
            {
                pin_parent_idx[pin_idx]
                for pin_idx in pin_list
                if pin_parent_idx[pin_idx] < benchmark.num_macros
            }
        )
        if len(parent_macros) >= 2:
            edge_weight = 1.0 / (len(parent_macros) - 1)
            for i in range(len(parent_macros)):
                for j in range(i + 1, len(parent_macros)):
                    pair = (parent_macros[i], parent_macros[j])
                    edge_dict[pair] = edge_dict.get(pair, 0.0) + edge_weight
                    adjacency_pairs[pair[0]][pair[1]] = (
                        adjacency_pairs[pair[0]].get(pair[1], 0.0) + edge_weight
                    )
                    adjacency_pairs[pair[1]][pair[0]] = (
                        adjacency_pairs[pair[1]].get(pair[0], 0.0) + edge_weight
                    )

    max_degree = max((len(pin_list) for pin_list in net_pin_lists), default=1)
    net_pins = torch.full((len(net_pin_lists), max_degree), -1, dtype=torch.long)
    net_mask = torch.zeros((len(net_pin_lists), max_degree), dtype=torch.bool)
    for net_idx, pin_list in enumerate(net_pin_lists):
        degree = len(pin_list)
        net_pins[net_idx, :degree] = torch.tensor(pin_list, dtype=torch.long)
        net_mask[net_idx, :degree] = True

    if edge_dict:
        macro_edges = torch.tensor(list(edge_dict.keys()), dtype=torch.long)
        macro_edge_weights = torch.tensor(list(edge_dict.values()), dtype=torch.float32)
    else:
        macro_edges = torch.zeros((0, 2), dtype=torch.long)
        macro_edge_weights = torch.zeros((0,), dtype=torch.float32)

    macro_adjacency: List[List[int]] = []
    macro_adjacency_weights: List[List[float]] = []
    for macro_idx in range(benchmark.num_macros):
        neighbors = adjacency_pairs[macro_idx]
        macro_adjacency.append(list(neighbors.keys()))
        macro_adjacency_weights.append(list(neighbors.values()))

    grid_width = float(benchmark.canvas_width / benchmark.grid_cols)
    grid_height = float(benchmark.canvas_height / benchmark.grid_rows)
    grid_x = torch.arange(benchmark.grid_cols, dtype=torch.float32)
    grid_y = torch.arange(benchmark.grid_rows, dtype=torch.float32)
    grid_xl = (grid_x * grid_width).repeat(benchmark.grid_rows)
    grid_xh = ((grid_x + 1.0) * grid_width).repeat(benchmark.grid_rows)
    grid_yl = torch.repeat_interleave(grid_y * grid_height, benchmark.grid_cols)
    grid_yh = torch.repeat_interleave((grid_y + 1.0) * grid_height, benchmark.grid_cols)

    return NetlistTensors(
        pin_parent_idx=torch.tensor(pin_parent_idx, dtype=torch.long, device=device),
        pin_offsets=torch.tensor(pin_offsets, dtype=torch.float32, device=device),
        net_pins=net_pins.to(device),
        net_mask=net_mask.to(device),
        net_weights=torch.tensor(net_weights, dtype=torch.float32, device=device),
        net_cnt=float(plc.net_cnt),
        macro_edges=macro_edges.to(device),
        macro_edge_weights=macro_edge_weights.to(device),
        macro_adjacency=macro_adjacency,
        macro_adjacency_weights=macro_adjacency_weights,
        grid_xl=grid_xl.to(device),
        grid_xh=grid_xh.to(device),
        grid_yl=grid_yl.to(device),
        grid_yh=grid_yh.to(device),
        grid_width=grid_width,
        grid_height=grid_height,
        grid_v_routes=grid_width * float(benchmark.vroutes_per_micron),
        grid_h_routes=grid_height * float(benchmark.hroutes_per_micron),
        vrouting_alloc=float(getattr(plc, "vrouting_alloc", 0.0)),
        hrouting_alloc=float(getattr(plc, "hrouting_alloc", 0.0)),
        smooth_range=int(getattr(plc, "smooth_range", 0)),
        congestion_scale=1.0,
    )
