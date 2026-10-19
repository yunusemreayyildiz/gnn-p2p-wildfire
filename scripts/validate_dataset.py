#!/usr/bin/env python3
"""Validate a dataset directory against the schemas generated from schemas/schemas.yaml.

Expected layout:
    <data_dir>/
        scenario_manifest.parquet
        fire_trace.parquet
        nodes/<deployment_id>.csv
        network_telemetry/<scenario_id>.parquet
        link_failure_labels/<scenario_id>.parquet      (optional)

Usage:
    python scripts/validate_dataset.py data_mock
    python scripts/validate_dataset.py data --strict        # warnings also fail

Exit code: 0 = clean, 1 = errors (or warnings with --strict).

Telemetry and labels are validated one scenario at a time, so memory use stays
bounded even for thousands of runs. Findings are aggregated per check: each check
prints how many scenarios/files hit it plus a few examples.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from generated.pyarrow_schemas import PRIMARY_KEYS, SCHEMA_VERSION, SCHEMAS  # noqa: E402

EPS = 1e-6


# --------------------------------------------------------------------------- report


class Report:
    def __init__(self, max_examples: int = 3):
        self.max_examples = max_examples
        self.issues: dict[tuple[str, str], dict] = {}
        self.info: list[str] = []

    def add(self, level: str, check: str, where: str, msg: str, rows: int = 1) -> None:
        d = self.issues.setdefault((level, check), {"count": 0, "rows": 0, "examples": []})
        d["count"] += 1
        d["rows"] += rows
        if len(d["examples"]) < self.max_examples:
            d["examples"].append(f"{where}: {msg}")

    def flag(self, level: str, check: str, where: str, bad: pd.DataFrame, msg: str, cols=()) -> None:
        """Record an issue if `bad` (the offending rows) is non-empty."""
        if bad is None or len(bad) == 0:
            return
        sample = ""
        if cols:
            sample = " e.g. " + str(bad.head(2)[list(cols)].to_dict("records"))
        self.add(level, check, where, f"{msg}{sample}", rows=len(bad))

    def total(self, level: str) -> int:
        return sum(d["count"] for (lvl, _), d in self.issues.items() if lvl == level)

    def checks(self, level: str) -> set[str]:
        return {c for (lvl, c) in self.issues if lvl == level}

    def print(self) -> None:
        for (level, check), d in sorted(self.issues.items(), key=lambda kv: (kv[0][0] != "error", kv[0][1])):
            tag = "ERROR" if level == "error" else "WARN "
            print(f"{tag} [{check}] {d['count']} occurrence(s), {d['rows']} row(s)")
            for ex in d["examples"]:
                print(f"        e.g. {ex}")
        for line in self.info:
            print(f"INFO  {line}")
        print(f"\n{self.total('error')} error occurrence(s) in {len(self.checks('error'))} check(s), "
              f"{self.total('warning')} warning occurrence(s) in {len(self.checks('warning'))} check(s)")


# ---------------------------------------------------------------------------- loading


def check_schema(rep: Report, name: str, table: pa.Table, src: str) -> bool:
    """Columns, types, nulls in required columns, enum values."""
    expected = SCHEMAS[name]
    ok = True
    have = set(table.schema.names)
    missing = [n for n in expected.names if n not in have]
    extra = [n for n in table.schema.names if n not in set(expected.names)]
    if missing:
        rep.add("error", "schema-missing-columns", src, str(missing))
        ok = False
    if extra:
        rep.add("error", "schema-extra-columns", src, str(extra))
        ok = False
    for f in expected:
        if f.name not in have:
            continue
        col = table.column(f.name)
        if col.type != f.type:
            rep.add("error", "schema-type-mismatch", src, f"{f.name}: {col.type} != expected {f.type}")
            ok = False
            continue
        if not f.nullable and col.null_count:
            rep.add("error", "null-in-required-column", src, f"{f.name}: {col.null_count} null(s)", rows=col.null_count)
            ok = False
        allowed = (f.metadata or {}).get(b"enum_values")
        if allowed is not None:
            allowed_set = set(allowed.decode().split(","))
            seen = {v for v in pc.unique(col).to_pylist() if v is not None}
            if seen - allowed_set:
                rep.add("error", "invalid-enum-value", src,
                        f"{f.name}: {sorted(seen - allowed_set)} not in {sorted(allowed_set)}")
                ok = False
    return ok


def read_parquet_checked(rep: Report, path: Path, name: str) -> pd.DataFrame | None:
    try:
        table = pq.read_table(path)
    except Exception as e:  # corrupt / unreadable file
        rep.add("error", "unreadable-file", path.name, str(e))
        return None
    if not check_schema(rep, name, table, path.name):
        return None
    return table.to_pandas()


def read_nodes_checked(rep: Report, path: Path) -> pd.DataFrame | None:
    schema = SCHEMAS["nodes"]
    opts = pacsv.ConvertOptions(column_types={f.name: f.type for f in schema}, strings_can_be_null=True)
    try:
        table = pacsv.read_csv(path, convert_options=opts)
    except Exception as e:
        rep.add("error", "unreadable-file", path.name, str(e))
        return None
    if not check_schema(rep, "nodes", table, path.name):
        return None
    return table.to_pandas()


def check_pk(rep: Report, name: str, df: pd.DataFrame, where: str) -> None:
    pk = PRIMARY_KEYS[name]
    rep.flag("error", "duplicate-primary-key", where, df[df.duplicated(pk, keep=False)], f"duplicate {pk}", pk)


def check_version(rep: Report, df: pd.DataFrame, where: str) -> None:
    rep.flag("error", "schema-version-mismatch", where, df[df["schema_version"] != SCHEMA_VERSION],
             f"schema_version != {SCHEMA_VERSION}", ["schema_version"])


# ------------------------------------------------------------------ table-level checks


def check_fire_trace(rep: Report, F: pd.DataFrame) -> None:
    w = "fire_trace"
    rep.flag("error", "fail-mode-vs-t-fail", w, F[F["t_fail_s"].isna() != (F["fail_mode"] == "none")],
             "t_fail_s must be null exactly when fail_mode == none", ["fire_id", "node_id", "fail_mode", "t_fail_s"])
    rep.flag("error", "fail-mode-vs-model", w,
             F[((F["failure_model"] == "fire") & (F["fail_mode"] == "random"))
               | ((F["failure_model"] == "random") & (F["fail_mode"] == "fire"))],
             "fail_mode contradicts failure_model", ["fire_id", "node_id", "failure_model", "fail_mode"])
    rep.flag("error", "random-has-fire-fields", w,
             F[(F["failure_model"] == "random") & (F["t_arrival_s"].notna() | F["flame_length_m"].notna())],
             "random failure rows must have null t_arrival_s / flame_length_m", ["fire_id", "node_id"])
    rep.flag("error", "fire-failure-without-arrival", w, F[(F["fail_mode"] == "fire") & F["t_arrival_s"].isna()],
             "fail_mode == fire requires t_arrival_s", ["fire_id", "node_id"])
    for col in ("t_arrival_s", "t_fail_s", "flame_length_m"):
        rep.flag("error", "negative-value", w, F[F[col] < 0], f"{col} < 0", ["fire_id", "node_id", col])


def check_failure_count_fairness(rep: Report, F: pd.DataFrame) -> None:
    """RQ1 fairness: the random-failure run should fail as many nodes as the fire run."""
    cnt = (F.assign(failed=F["t_fail_s"].notna())
            .groupby(["fire_id", "deployment_id", "failure_model"])["failed"].sum().unstack("failure_model"))
    if {"fire", "random"} <= set(cnt.columns):
        both = cnt.dropna(subset=["fire", "random"])
        for (fid, dep), r in both[both["fire"] != both["random"]].iterrows():
            rep.add("warning", "fire-vs-random-failure-count", f"{fid}/{dep}",
                    f"{int(r['fire'])} fire failures vs {int(r['random'])} random failures")


# -------------------------------------------------------------- scenario-level checks


def check_telemetry(rep: Report, m: pd.Series, T: pd.DataFrame, nd: pd.DataFrame,
                    fd: pd.DataFrame | None) -> pd.DataFrame:
    """Validate one scenario's telemetry. Returns the cleaned, sorted frame for label checks."""
    sid = m["scenario_id"]
    known = set(nd["node_id"])

    rep.flag("error", "scenario-id-mismatch", sid, T[T["scenario_id"] != sid], "scenario_id differs from file name")
    rep.flag("error", "seed-mismatch", sid, T[T["seed"] != m["seed"]], "seed differs from manifest", ["seed"])
    check_version(rep, T, sid)
    rep.flag("error", "unknown-node", sid, T[~T["node_id"].isin(known)], "node_id not in nodes.csv", ["node_id"])
    T = T[T["node_id"].isin(known)]

    dup = T[T.duplicated(["node_id", "t"], keep=False)]
    rep.flag("error", "duplicate-primary-key", sid, dup, "duplicate (node_id, t)", ["node_id", "t"])
    T = T.drop_duplicates(["node_id", "t"])

    missing = known - set(T["node_id"])
    if missing:
        rep.add("error", "missing-nodes", sid, f"no telemetry for nodes {sorted(missing)}")
    if T.groupby("node_id").size().nunique() > 1:
        rep.add("error", "irregular-grid", sid, "nodes have different numbers of rows")

    T = T.sort_values(["node_id", "t"]).reset_index(drop=True)
    g = T.groupby("node_id", sort=False)

    # time axis
    rep.flag("error", "negative-value", sid, T[T["t"] < 0], "t < 0", ["node_id", "t"])
    dt = g["t"].diff()
    rep.flag("warning", "time-step-vs-window", sid, T[dt.notna() & ((dt - T["window_s"]).abs() > EPS)],
             "time step differs from window_s", ["node_id", "t", "window_s"])
    if T["window_s"].nunique() > 1:
        rep.add("warning", "window-not-constant", sid, f"window_s values {sorted(T['window_s'].unique())[:5]}")

    # value ranges
    for col in ["battery_j", "tx_energy_j", "pkts_tx", "pkts_rx", "pkts_fwd", "pkts_drop", "queue_len",
                "n_neighbors", "latency_mean_s", "hop_to_sink"]:
        rep.flag("error", "negative-value", sid, T[T[col] < 0], f"{col} < 0", ["node_id", "t", col])
    rep.flag("error", "non-positive-window", sid, T[T["window_s"] <= 0], "window_s <= 0", ["node_id", "t"])
    rep.flag("error", "pdr-out-of-range", sid, T[(T["pdr_window"] < 0) | (T["pdr_window"] > 1)],
             "pdr_window outside [0, 1]", ["node_id", "t", "pdr_window"])
    rep.flag("error", "fwd-exceeds-tx", sid, T[T["pkts_fwd"] > T["pkts_tx"]],
             "pkts_fwd > pkts_tx (tx includes forwarded packets)", ["node_id", "t", "pkts_fwd", "pkts_tx"])
    rep.flag("error", "battery-increased", sid, T[g["battery_j"].diff() > EPS],
             "battery_j increased between samples", ["node_id", "t", "battery_j"])

    # routing fields
    rep.flag("error", "bad-parent", sid, T[T["parent_id"].notna() & ~T["parent_id"].isin(known)],
             "parent_id not in nodes.csv", ["node_id", "t", "parent_id"])
    rep.flag("error", "self-parent", sid, T[T["parent_id"] == T["node_id"]], "parent_id == node_id", ["node_id", "t"])

    # liveness
    dead = (~T["is_alive"]).astype(int)
    ever_dead = dead.groupby(T["node_id"]).cummax().astype(bool)
    prev_dead = ever_dead.astype(int).groupby(T["node_id"]).shift(1, fill_value=0).astype(bool)
    rep.flag("error", "resurrected-node", sid, T[T["is_alive"] & ever_dead], "is_alive went False -> True",
             ["node_id", "t"])
    rep.flag("error", "tx-after-death", sid, T[prev_dead & (T["pkts_tx"] > 0)],
             "dead node still transmits", ["node_id", "t", "pkts_tx"])

    # consistency with fire_trace
    if fd is not None and len(fd):
        t_fail = T["node_id"].map(fd.set_index("node_id")["t_fail_s"])
        after = t_fail.notna() & (T["t"] >= t_fail)
        rep.flag("error", "alive-after-failure", sid, T[after & T["is_alive"]],
                 "is_alive True at/after fire_trace.t_fail_s", ["node_id", "t"])
        rep.flag("warning", "dead-without-cause", sid, T[~T["is_alive"] & ~after & (T["battery_j"] > EPS)],
                 "node dead before t_fail_s with battery left", ["node_id", "t", "battery_j"])
    return T


def check_labels(rep: Report, m: pd.Series, L: pd.DataFrame, T: pd.DataFrame,
                 fd: pd.DataFrame | None) -> None:
    sid = m["scenario_id"]
    rep.flag("error", "scenario-id-mismatch", sid, L[L["scenario_id"] != sid], "label scenario_id differs from file name")
    check_pk(rep, "link_failure_labels", L, sid)

    merged = L.merge(T[["node_id", "t"]], on=["node_id", "t"], how="left", indicator=True)
    rep.flag("error", "label-without-telemetry", sid, merged[merged["_merge"] == "left_only"],
             "label key (node_id, t) not in telemetry", ["node_id", "t"])

    td = L["t_disconnect_s"]
    expected = td.notna() & (L["t"] < td) & (td <= L["t"] + L["horizon_s"] + EPS)
    rep.flag("error", "label-inconsistent", sid, L[expected != L["y_disconnect"]],
             "y_disconnect != (t < t_disconnect_s <= t + horizon_s)",
             ["node_id", "t", "horizon_s", "y_disconnect", "t_disconnect_s"])
    rep.flag("error", "cause-vs-t-disconnect", sid, L[(L["cause"] == "none") != td.isna()],
             "cause == none must hold exactly when t_disconnect_s is null", ["node_id", "t", "cause"])
    rep.flag("warning", "label-after-disconnect", sid, L[td.notna() & (L["t"] >= td)],
             "label row at/after the disconnection time", ["node_id", "t", "t_disconnect_s"])

    if fd is not None and len(fd):
        mode = L["node_id"].map(fd.set_index("node_id")["fail_mode"])
        for cause in ("fire", "random"):
            rep.flag("error", "cause-vs-fire-trace", sid, L[(L["cause"] == cause) & (mode != cause)],
                     f"cause == {cause} but fire_trace.fail_mode differs", ["node_id", "t", "cause"])


# ------------------------------------------------------------------------ orchestrator


def validate(data_dir: Path, rep: Report) -> None:
    data_dir = Path(data_dir)
    mpath, fpath = data_dir / "scenario_manifest.parquet", data_dir / "fire_trace.parquet"
    ndir, tdir, ldir = data_dir / "nodes", data_dir / "network_telemetry", data_dir / "link_failure_labels"

    fatal = False
    for p in (mpath, fpath):
        if not p.exists():
            rep.add("error", "missing-file", p.name, "required file not found")
            fatal = True
    for d in (ndir, tdir):
        if not d.is_dir():
            rep.add("error", "missing-file", d.name + "/", "required directory not found")
            fatal = True
    if fatal:
        return

    # ---- manifest
    M = read_parquet_checked(rep, mpath, "scenario_manifest")
    if M is None:
        return
    check_pk(rep, "scenario_manifest", M, "scenario_manifest")
    check_version(rep, M, "scenario_manifest")

    split_n = M.groupby("fire_id")["split"].nunique()
    for fid in split_n[split_n > 1].index:
        rep.add("error", "split-leakage", fid,
                f"fire_id appears in splits {sorted(M.loc[M['fire_id'] == fid, 'split'].unique())}")

    # ---- nodes
    deployments: dict[str, pd.DataFrame] = {}
    for p in sorted(ndir.glob("*.csv")):
        df = read_nodes_checked(rep, p)
        if df is None:
            continue
        rep.flag("error", "deployment-id-mismatch", p.name, df[df["deployment_id"] != p.stem],
                 "deployment_id column differs from file name")
        if not (df["role"] == "sink").any():
            rep.add("error", "no-sink", p.name, "deployment has no sink node")
        deployments[p.stem] = df
    N = pd.concat(deployments.values(), ignore_index=True) if deployments else pd.DataFrame(columns=["deployment_id", "node_id"])
    check_pk(rep, "nodes", N, "nodes")
    for dep in sorted(set(M["deployment_id"]) - set(deployments)):
        rep.add("error", "unknown-deployment", dep, "manifest references a deployment without a valid nodes file")
    for dep in sorted(set(deployments) - set(M["deployment_id"])):
        rep.add("warning", "unused-deployment", dep, "nodes file not referenced by the manifest")

    # ---- fire trace
    F = read_parquet_checked(rep, fpath, "fire_trace")
    fgroups: dict[tuple, pd.DataFrame] = {}
    if F is not None:
        check_pk(rep, "fire_trace", F, "fire_trace")
        check_fire_trace(rep, F)
        check_failure_count_fairness(rep, F)
        key = ["fire_id", "deployment_id", "failure_model", "node_id"]
        need = (M[["fire_id", "deployment_id", "failure_model"]].drop_duplicates()
                .merge(N[["deployment_id", "node_id"]], on="deployment_id"))
        cmp = need.merge(F[key], on=key, how="outer", indicator=True)
        rep.flag("error", "fire-trace-missing", "fire_trace", cmp[cmp["_merge"] == "left_only"],
                 "manifest x nodes combination has no fire_trace row", key)
        rep.flag("error", "fire-trace-extra", "fire_trace", cmp[cmp["_merge"] == "right_only"],
                 "fire_trace row matches no manifest scenario / node", key)
        fgroups = {k: g for k, g in F.groupby(["fire_id", "deployment_id", "failure_model"])}

    # ---- per-scenario telemetry + labels
    manifest_ids = set(M["scenario_id"])
    for stem in sorted(p.stem for p in tdir.glob("*.parquet")):
        if stem not in manifest_ids:
            rep.add("error", "orphan-telemetry-file", stem, "telemetry file has no manifest row")

    has_labels_dir = ldir.is_dir()
    if not has_labels_dir:
        rep.info.append("link_failure_labels/ not found; label checks skipped")
    n_tele_rows = 0
    label_stats: dict[tuple[str, float], list[int]] = {}

    for _, m in M.iterrows():
        sid = m["scenario_id"]
        nd = deployments.get(m["deployment_id"])
        if nd is None:
            continue
        tp = tdir / f"{sid}.parquet"
        if not tp.exists():
            rep.add("error", "missing-telemetry", sid, "no telemetry file")
            continue
        T = read_parquet_checked(rep, tp, "network_telemetry")
        if T is None:
            continue
        n_tele_rows += len(T)
        fd = fgroups.get((m["fire_id"], m["deployment_id"], m["failure_model"]))
        T = check_telemetry(rep, m, T, nd, fd)

        if has_labels_dir:
            lp = ldir / f"{sid}.parquet"
            if not lp.exists():
                rep.add("warning", "missing-labels", sid, "no label file")
                continue
            L = read_parquet_checked(rep, lp, "link_failure_labels")
            if L is None:
                continue
            check_labels(rep, m, L, T, fd)
            for h, grp in L.groupby("horizon_s"):
                s = label_stats.setdefault((m["split"], float(h)), [0, 0])
                s[0] += int(grp["y_disconnect"].sum())
                s[1] += len(grp)

    # ---- summary
    rep.info.append(f"{len(M)} scenario(s), {M['fire_id'].nunique()} fire(s), {len(deployments)} deployment(s), "
                    f"{n_tele_rows} telemetry row(s)")
    rep.info.append("scenarios per split: " + ", ".join(f"{k}={v}" for k, v in M["split"].value_counts().sort_index().items()))
    for (split, h), (pos, n) in sorted(label_stats.items()):
        rep.info.append(f"labels split={split} horizon={h:g}s: {pos}/{n} positive ({100 * pos / n:.1f}%)" if n else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--strict", action="store_true", help="treat warnings as failures")
    ap.add_argument("--max-examples", type=int, default=3)
    args = ap.parse_args(argv)

    rep = Report(max_examples=args.max_examples)
    validate(args.data_dir, rep)
    rep.print()
    failed = rep.total("error") > 0 or (args.strict and rep.total("warning") > 0)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
