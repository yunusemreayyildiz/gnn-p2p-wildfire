"""Tests for scripts/validate_dataset.py. Run: pytest tests/test_validate_dataset.py"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scripts.make_mock_dataset as make_mock_dataset  # noqa: E402
import scripts.validate_dataset as vd  # noqa: E402
from generated.pyarrow_schemas import SCHEMAS  # noqa: E402

FIRE = "F01_D1_tree_fire_s1"
RAND = "F01_D1_tree_random_s1"


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["make_mock_dataset.py", str(tmp_path)])
    make_mock_dataset.main()
    return tmp_path


def run(data_dir):
    rep = vd.Report()
    vd.validate(data_dir, rep)
    return rep


def mutate(data_dir, rel, table_name, fn):
    """Read a parquet file, apply fn(DataFrame) -> DataFrame, write it back with the schema."""
    path = data_dir / rel
    df = pq.read_table(path).to_pandas()
    df = fn(df)
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMAS[table_name], preserve_index=False), path)


def test_clean_mock_passes(data):
    rep = run(data)
    assert rep.issues == {}
    assert vd.main([str(data)]) == 0


def test_duplicate_telemetry_row(data):
    mutate(data, f"network_telemetry/{FIRE}.parquet", "network_telemetry", lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True))
    assert "duplicate-primary-key" in run(data).checks("error")


def test_resurrected_node_and_alive_after_failure(data):
    def f(d):
        d.loc[(d.node_id == 1) & (d.t == 600.0), "is_alive"] = True
        return d
    mutate(data, f"network_telemetry/{FIRE}.parquet", "network_telemetry", f)
    errs = run(data).checks("error")
    assert {"resurrected-node", "alive-after-failure"} <= errs


def test_battery_increase(data):
    def f(d):
        d.loc[(d.node_id == 2) & (d.t == 300.0), "battery_j"] += 50
        return d
    mutate(data, f"network_telemetry/{FIRE}.parquet", "network_telemetry", f)
    assert "battery-increased" in run(data).checks("error")


def test_split_leakage(data):
    def f(d):
        d.loc[d.scenario_id == RAND, "split"] = "test"   # F01 now in train and test
        return d
    mutate(data, "scenario_manifest.parquet", "scenario_manifest", f)
    assert "split-leakage" in run(data).checks("error")


def test_label_inconsistent(data):
    def f(d):
        i = d.index[(d.node_id == 1) & (d.t == 240.0) & (d.horizon_s == 120.0)][0]
        d.loc[i, "y_disconnect"] = not d.loc[i, "y_disconnect"]
        return d
    mutate(data, f"link_failure_labels/{FIRE}.parquet", "link_failure_labels", f)
    assert "label-inconsistent" in run(data).checks("error")


def test_cause_contradicts_fire_trace(data):
    def f(d):
        d.loc[d.node_id == 1, "cause"] = "fire"          # but this is the random-failure run
        return d
    mutate(data, f"link_failure_labels/{RAND}.parquet", "link_failure_labels", f)
    assert "cause-vs-fire-trace" in run(data).checks("error")


def test_fire_info_leaking_into_telemetry_is_rejected(data):
    path = data / f"network_telemetry/{FIRE}.parquet"
    table = pq.read_table(path)
    table = table.append_column("flame_length_m", pa.array([1.0] * table.num_rows, pa.float32()))
    pq.write_table(table, path)
    assert "schema-extra-columns" in run(data).checks("error")


def test_invalid_enum_value(data):
    def f(d):
        d.loc[0, "protocol"] = "gossip"
        return d
    mutate(data, "scenario_manifest.parquet", "scenario_manifest", f)
    assert "invalid-enum-value" in run(data).checks("error")


def test_wrong_column_type(data):
    t = pq.read_table(data / "fire_trace.parquet")
    t = t.set_column(t.schema.get_field_index("node_id"), "node_id", t.column("node_id").cast(pa.int64()))
    pq.write_table(t, data / "fire_trace.parquet")
    assert "schema-type-mismatch" in run(data).checks("error")


def test_missing_fire_trace_rows(data):
    mutate(data, "fire_trace.parquet", "fire_trace", lambda d: d[~((d.fire_id == "F02") & (d.node_id == 2))])
    assert "fire-trace-missing" in run(data).checks("error")


def test_random_row_with_fire_fields(data):
    def f(d):
        d.loc[(d.failure_model == "random") & (d.node_id == 1), "t_arrival_s"] = 400.0
        return d
    mutate(data, "fire_trace.parquet", "fire_trace", f)
    assert "random-has-fire-fields" in run(data).checks("error")


def test_orphan_and_missing_files(data):
    (data / f"network_telemetry/{RAND}.parquet").rename(data / "network_telemetry/GHOST.parquet")
    errs = run(data).checks("error")
    assert {"orphan-telemetry-file", "missing-telemetry"} <= errs


def test_failure_count_mismatch_is_reported(data):
    def f(d):
        d.loc[(d.failure_model == "random") & (d.node_id == 1), ["t_fail_s", "fail_mode"]] = [None, "none"]
        return d
    mutate(data, "fire_trace.parquet", "fire_trace", f)
    assert "fire-vs-random-failure-count" in run(data).checks("warning")


def test_warnings_pass_normally_and_fail_with_strict(data):
    text = (data / "nodes/D1.csv").read_text().replace("D1,", "D9,")
    (data / "nodes/D9.csv").write_text(text)          # valid file, but no scenario uses it -> warning only
    rep = run(data)
    assert rep.checks("warning") == {"unused-deployment"} and not rep.checks("error")
    assert vd.main([str(data)]) == 0
    assert vd.main([str(data), "--strict"]) == 1
