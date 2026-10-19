#!/usr/bin/env python3
"""Generate a tiny, internally consistent mock dataset (Week 1-2 end-to-end join test).

Topology: node 0 = sink, node 1 = sensor (parent 0), node 2 = sensor (parent 1).
Scenarios:
    F01_D1_tree_fire_s1    fire kills node 1 at t=300; node 2 loses its route at t=300 (topology)
    F01_D1_tree_random_s1  random failure kills node 1 at t=420 (same failure count as the fire run)
    F02_D1_tree_fire_s1    fire never reaches the network; no failures

Usage: python scripts/make_mock_dataset.py data_mock
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from generated.pyarrow_schemas import SCHEMA_VERSION, SCHEMAS  # noqa: E402

DEPLOYMENT = "D1"
HORIZONS = (120.0, 240.0)
N_STEPS, WINDOW = 10, 60.0
INF = float("inf")

# (scenario_id, fire_id, failure_model, split, seed, failure_time_of_node_1)
SCENARIOS = [
    ("F01_D1_tree_fire_s1", "F01", "fire", "train", 1, 300.0),
    ("F01_D1_tree_random_s1", "F01", "random", "train", 2, 420.0),
    ("F02_D1_tree_fire_s1", "F02", "fire", "test", 3, INF),
]


def write(rows: list[dict], name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), path)


def nodes_rows() -> list[dict]:
    spec = [(0, 500000.0, 4100000.0, 120.0, "sink", "GR2"),
            (1, 500800.0, 4100200.0, 180.0, "sensor", "TU1"),
            (2, 501600.0, 4100500.0, 240.0, "sensor", "TL3")]
    return [dict(deployment_id=DEPLOYMENT, node_id=n, x=x, y=y, elevation_m=z, role=r, fuel_model=f,
                 tx_power_dbm=14.0, battery_init_j=2000.0, sf_init=9) for n, x, y, z, r, f in spec]


def fire_trace_rows() -> list[dict]:
    def row(fire, model, node, arr, flame, fail, mode):
        return dict(fire_id=fire, deployment_id=DEPLOYMENT, failure_model=model, node_id=node,
                    t_arrival_s=arr, flame_length_m=flame, t_fail_s=fail, fail_mode=mode)
    rows = []
    for node in (0, 1, 2):
        # F01 fire: only node 1 burns (arrives 280, fails 300)
        burned = node == 1
        rows.append(row("F01", "fire", node, 280.0 if burned else None, 3.2 if burned else None,
                        300.0 if burned else None, "fire" if burned else "none"))
        # F01 random: matched failure count (1), different node time
        rows.append(row("F01", "random", node, None, None, 420.0 if burned else None,
                        "random" if burned else "none"))
        # F02 fire: never reaches the network
        rows.append(row("F02", "fire", node, None, None, None, "none"))
    return rows


def telemetry_rows(sid: str, seed: int, fail_time: float, rng: random.Random) -> list[dict]:
    rows = []
    battery = {0: 2000.0, 1: 2000.0, 2: 2000.0}
    for k in range(1, N_STEPS + 1):
        t = WINDOW * k
        n1_alive = t < fail_time
        for node in (0, 1, 2):
            alive = n1_alive if node == 1 else True
            if node == 0:     # sink
                v = dict(parent_id=None, hop_to_sink=0, n_neighbors=int(n1_alive), tx=0, rx=8 if n1_alive else 0,
                         fwd=0, pdr=None, lat=2.0 if n1_alive else None, e=0.0)
            elif node == 1 and alive:
                v = dict(parent_id=0, hop_to_sink=1, n_neighbors=2, tx=10, rx=6, fwd=5, pdr=0.95, lat=1.1, e=0.8)
            elif node == 1:  # dead
                v = dict(parent_id=None, hop_to_sink=None, n_neighbors=0, tx=0, rx=0, fwd=0, pdr=None, lat=None, e=0.0)
            elif n1_alive:    # node 2 with route
                v = dict(parent_id=1, hop_to_sink=2, n_neighbors=1, tx=5, rx=0, fwd=0, pdr=0.9, lat=2.2, e=0.4)
            else:             # node 2 lost its route
                v = dict(parent_id=None, hop_to_sink=None, n_neighbors=0, tx=3, rx=0, fwd=0, pdr=0.0, lat=None, e=0.3)
            battery[node] -= v["e"]
            heard = v["n_neighbors"] > 0
            rows.append(dict(
                schema_version=SCHEMA_VERSION, scenario_id=sid, seed=seed, t=t, node_id=node, window_s=WINDOW,
                is_alive=alive, battery_j=round(battery[node], 3), tx_energy_j=v["e"],
                parent_id=v["parent_id"], hop_to_sink=v["hop_to_sink"], n_neighbors=v["n_neighbors"],
                rssi_mean_dbm=round(-100 + rng.gauss(0, 2), 2) if heard else None,
                snr_mean_db=round(6 + rng.gauss(0, 1), 2) if heard else None,
                pdr_window=v["pdr"], pkts_tx=v["tx"], pkts_rx=v["rx"], pkts_fwd=v["fwd"], pkts_drop=0,
                queue_len=0, latency_mean_s=v["lat"]))
    return rows


def label_rows(sid: str, model: str, fail_time: float) -> list[dict]:
    rows = []
    for node in (1, 2):                       # sensors only
        if fail_time == INF:
            t_disc, cause = None, "none"
        else:
            t_disc = fail_time
            cause = model if node == 1 else "topology"   # node 1 dies by fire/random, node 2 loses its parent
        for k in range(1, N_STEPS + 1):
            t = WINDOW * k
            if t_disc is not None and t >= t_disc:
                continue                       # only label rows that are still connected
            for h in HORIZONS:
                y = t_disc is not None and t < t_disc <= t + h
                rows.append(dict(scenario_id=sid, node_id=node, t=t, horizon_s=h, y_disconnect=bool(y),
                                 t_disconnect_s=t_disc, cause=cause))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    out = ap.parse_args().out

    manifest = [dict(scenario_id=sid, fire_id=fire, deployment_id=DEPLOYMENT, protocol="tree", failure_model=model,
                     seed=seed, fire_seed=None, split=split, code_commit_fire_model="mock",
                     code_commit_network_model="mock", schema_version=SCHEMA_VERSION)
                for sid, fire, model, split, seed, _ in SCENARIOS]
    write(manifest, "scenario_manifest", out / "scenario_manifest.parquet")
    write(fire_trace_rows(), "fire_trace", out / "fire_trace.parquet")

    (out / "nodes").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nodes_rows()).to_csv(out / "nodes" / f"{DEPLOYMENT}.csv", index=False)

    for sid, _, model, _, seed, fail_time in SCENARIOS:
        write(telemetry_rows(sid, seed, fail_time, random.Random(seed)), "network_telemetry",
              out / "network_telemetry" / f"{sid}.parquet")
        write(label_rows(sid, model, fail_time), "link_failure_labels",
              out / "link_failure_labels" / f"{sid}.parquet")
    print(f"mock dataset written to {out}")


if __name__ == "__main__":
    main()
