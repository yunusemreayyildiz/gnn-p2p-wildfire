# Fire-Aware Sensor Network Resilience Testbed
Routing resilience and early link-failure prediction with GNNs in wildfire-affected sensor networks: a simulation study for a pilot region in Mugla, Turkiye.

> **Status:** early development (Weeks 1-2: interface schemas, mock data, feasibility study). Interfaces and schemas may change until `v0.1-schema` is tagged and the open decisions below are closed.

## Overview

Sensor networks are a fast and cheap way to detect forest fires early, but a fire also destroys the network that is supposed to report it: as the fire front advances, nodes fail, routes break and alarm packets are lost. Fire cannot be applied to a real network for experiments, so this project builds a **reproducible simulation testbed**.

The testbed is not called a "digital twin": it has no bidirectional link to a physical system.

### Research questions

- **RQ1.** As nodes are lost to the advancing fire front, how do different routing / clustering strategies (flooding, tree-based routing, HEED-like clustering) affect delivery ratio and network lifetime? How much do the results differ from a random-failure model?
- **RQ2.** Without any fire information, how early and how accurately can a GNN predict node or link disconnection from network telemetry alone, compared with classical baselines (persistence, GBM, LSTM, graph connectivity analysis)?

### Intended contributions

1. A physics-based (Rothermel) fire model validated quantitatively against a real fire, with node failures derived from it.
2. Measurement of how fire-driven failure differs from random failure.
3. GNN-based disconnection prediction from network telemetry only, compared against strong baselines.
4. An open, reproducible scenario dataset.

## Architecture

Three components communicate through file-based interfaces. Every file carries a `scenario_id` / seed so any result can be traced back and reproduced.

```
Static data                    Fire model (Python)                 Network model                     Analytics (PyTorch Geometric)
Sentinel-2 (NDMI), DEM,   -->  Rothermel + elliptical cellular --> OMNeT++ / INET-FLoRa          -->  GraphSAGE / GAT + GRU
CORINE -> Scott & Burgan,      automaton                           (fallback: SimPy)                    vs. persistence / GBM / LSTM /
ERA5 / Open-Meteo              -> fire_trace                       fire_trace + nodes.csv               connectivity baselines
                               validation: burned-area IoU,        -> network_telemetry                 -> metrics: PR-AUC, F1, lead time,
                               arrival-time error                                                       false-alarm rate
```

**Core scope:** the three components and the RQ1-RQ2 experiments.
**Optional (only if the core is complete):** priority alarm-packet analysis, a simple map UI, an independently labelled (FWI) risk score, and a Kafka-based replay / online-inference demo (see ADR-001, planned in `docs/adr/`).
**Out of scope:** Q-learning fire spread, live co-simulation, hardware.

The GNN input is **network telemetry only**. No fire-derived feature (flame length, distance to fire front, ...) appears in `network_telemetry`; the schema enforces this.

## Repository layout

```
schemas/schemas.yaml           Single source of truth for all dataset schemas
generated/                     PyArrow schemas and Pydantic models (generated, do not edit)
scripts/gen_schemas.py         YAML -> PyArrow + Pydantic generator (--check for CI)
scripts/validate_dataset.py    Dataset validator (schema, keys, leakage, consistency)
scripts/make_mock_dataset.py   Tiny mock dataset for end-to-end tests
tests/                         Schema, generator and validator tests
.github/workflows/ci.yml       CI: generated files up to date + tests

# planned
fire_model/                    Fire and environment model (Python)
network_model/                 OMNeT++ / SimPy network model
ml/                            Baselines and GNN experiments
docs/                          Project proposal, ADRs, interface notes
```

## Getting started

Requires Python 3.13 (CI version; other recent 3.x versions are likely fine).

```bash
pip install -r requirements.txt

# regenerate PyArrow schemas and Pydantic models after editing schemas/schemas.yaml
python scripts/gen_schemas.py

# generate and validate a small mock dataset
python scripts/make_mock_dataset.py data_mock
python scripts/validate_dataset.py data_mock

# run the tests
python -m pytest tests -q
```

## Data

Datasets are not stored in this repository (they are git-ignored). Layout expected by the validator:

```
<data_dir>/
    scenario_manifest.parquet            one row per simulation run
    fire_trace.parquet                   per-node fire arrival / failure times
    nodes/<deployment_id>.csv            static node properties
    network_telemetry/<scenario_id>.parquet
    link_failure_labels/<scenario_id>.parquet    (optional) prediction targets
```

| Table | Key | Purpose |
|---|---|---|
| `scenario_manifest` | `scenario_id` | Links fire, deployment, protocol, failure model, seed and train/val/test split |
| `nodes` | `deployment_id, node_id` | Position (single dataset CRS), elevation, role, radio and energy parameters |
| `fire_trace` | `fire_id, deployment_id, failure_model, node_id` | Fire arrival time, flame length, failure time and mode |
| `network_telemetry` | `scenario_id, node_id, t` | Per-node network state per time window (RSSI/SNR, PDR, energy, routing state, counters) |
| `link_failure_labels` | `scenario_id, node_id, t, horizon_s` | Disconnection labels and lead-time reference, kept separate to prevent feature/label leakage |

Conventions: all times are in seconds from fire ignition; the train/val/test split is assigned at the `fire_id` level so the same terrain/wind combination never appears in both train and test. See [`schemas/schemas.yaml`](schemas/schemas.yaml) for every field, type and unit.

### Validation

`scripts/validate_dataset.py` checks schema conformance, primary-key uniqueness, referential integrity across tables, split leakage, physical consistency (battery, liveness, packet counters), label consistency, and fire-vs-random failure-count matching. It exits with code 1 on errors (`--strict` also fails on warnings). **Simulator output should pass the validator before it is used for training.**

## Contributing

- `main` is protected and must always be green. Work on short-lived branches (`feat/...`, `schema/...`) and merge by pull request.
- Changing `schemas/schemas.yaml` requires regenerating `generated/` in the same PR (CI runs `gen_schemas.py --check`) and review by all team members.
- Never edit files in `generated/` by hand.
- Tags mark milestones (`v0.1-schema`, interim report, dataset v1, final) so every reported result maps to a code version.

## Open decisions

| Decision | Affects |
|---|---|
| Disconnection definition (node-level sink reachability vs. link-level) | Labels, metrics |
| Prediction horizon `H` (single value or several) | Labels, lead-time metric |
| Failure time definition (`t_fail_s` vs. arrival time, radiant-heat margin) | `fire_trace`, RQ1 |
| Random-failure matching method and experiment grid size (about 1350 vs. 2700 runs) | RQ1, compute budget |
| Telemetry sampling interval (LoRa duty-cycle constraints) | `network_telemetry` |

Markers `TODO(decision)` in `schemas/schemas.yaml` track these.

## Team

| Role | Member |
|---|---|
| Fire model and environment data | TBD |
| OMNeT++ network model | TBD |
| ML and data pipeline | TBD |

Supervisor: TBD

## License

TBD (to be decided before the open-source release at the end of the project).