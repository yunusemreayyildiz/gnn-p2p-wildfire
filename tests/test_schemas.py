"""Smoke tests for generated schemas. Run: pytest tests/test_schemas.py"""
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generated.models import MODELS, FireTrace, NetworkTelemetry  # noqa: E402
from generated.pyarrow_schemas import KAFKA, PRIMARY_KEYS, SCHEMAS  # noqa: E402


def telemetry_row(**over):
    row = dict(
        schema_version="0.1", scenario_id="F01_D1_S1_tree_R1", seed=1, t=60.0, node_id=3,
        window_s=60.0, is_alive=True, battery_j=1200.5, tx_energy_j=0.8, parent_id=1,
        hop_to_sink=2, n_neighbors=4, rssi_mean_dbm=-101.2, snr_mean_db=5.5, pdr_window=0.93,
        pkts_tx=10, pkts_rx=12, pkts_fwd=5, pkts_drop=1, queue_len=0, latency_mean_s=2.1,
    )
    row.update(over)
    return row


def test_every_entity_has_model_and_schema():
    assert set(MODELS) == set(SCHEMAS)


def test_model_fields_match_arrow_fields():
    for name, model in MODELS.items():
        assert list(model.model_fields) == SCHEMAS[name].names, name


def test_valid_telemetry_row():
    NetworkTelemetry(**telemetry_row())


def test_nullable_fields_accept_none_but_are_required():
    NetworkTelemetry(**telemetry_row(parent_id=None, rssi_mean_dbm=None, pdr_window=None))
    row = telemetry_row()
    del row["parent_id"]
    with pytest.raises(ValidationError):
        NetworkTelemetry(**row)  # missing key is an error even if nullable


def test_non_nullable_rejects_none():
    with pytest.raises(ValidationError):
        NetworkTelemetry(**telemetry_row(battery_j=None))


def test_ratio_constraint_and_extra_fields_and_enum():
    with pytest.raises(ValidationError):
        NetworkTelemetry(**telemetry_row(pdr_window=1.2))
    with pytest.raises(ValidationError):
        NetworkTelemetry(**telemetry_row(flame_length_m=3.0))  # fire info must not leak into telemetry
    with pytest.raises(ValidationError):
        FireTrace(fire_id="F01", deployment_id="D1", failure_model="fire", node_id=1,
                  t_arrival_s=None, flame_length_m=None, t_fail_s=None, fail_mode="lightning")


def test_parquet_roundtrip_preserves_schema(tmp_path):
    rows = [NetworkTelemetry(**telemetry_row(node_id=i)).model_dump() for i in range(3)]
    table = pa.Table.from_pylist(rows, schema=SCHEMAS["network_telemetry"])
    path = tmp_path / "t.parquet"
    pq.write_table(table, path)
    back = pq.read_table(path)
    assert back.schema.equals(SCHEMAS["network_telemetry"], check_metadata=False)
    assert back.num_rows == 3


def test_primary_keys_and_kafka_metadata():
    assert PRIMARY_KEYS["fire_trace"] == ["fire_id", "deployment_id", "failure_model", "node_id"]
    assert KAFKA["network_telemetry"]["topic"] == "telemetry.raw"
    assert KAFKA["predictions_link_failure"]["topic"] == "predictions.link_failure"


def test_protected_namespace_field_allowed():
    assert "model_version" in MODELS["predictions_link_failure"].model_fields


def test_generated_files_are_fresh():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_schemas.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
