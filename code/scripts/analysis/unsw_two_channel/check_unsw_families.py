#!/usr/bin/env python
"""Structural check only: derive UNSW feature families by the frozen rule.

Produces no judgment-bearing numbers -- only column lists and group sizes.
Applies PROTOCOL_TWO_CHANNEL_20260903.md 2.1 mechanically, twice:
  (a) META including window_start_epoch  -> the correct, leak-free version
  (b) META excluding window_start_epoch  -> reproduces the 62-column claim
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

REPO = Path.home() / "iot-device-classification"
DAY_CSV = REPO / "results/unsw_features_full/features_day_16-09-23.csv"

META_WITH = {
    "device", "day", "label", "source_file",
    "window_id", "window_start", "window_end", "window_start_epoch",
}
META_WITHOUT = META_WITH - {"window_start_epoch"}


def feature_columns(df, meta):
    return [c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def derive_families(cols):
    groups = OrderedDict()
    for c in cols:
        groups.setdefault(c.split("_")[0], []).append(c)
    kept, singles = {}, []
    for k, v in groups.items():
        if len(v) >= 2:
            kept[k] = v
        else:
            singles.extend(v)
    if singles:
        kept["singletons"] = singles
    return OrderedDict(sorted(kept.items(), key=lambda kv: (-len(kv[1]), kv[0])))


def report(tag, df, meta):
    cols = feature_columns(df, meta)
    fams = derive_families(cols)
    print(f"[{tag}] feature columns = {len(cols)}")
    print(f"[{tag}] groups = " + " ".join(f"{k}{len(v)}" for k, v in fams.items()))
    print(f"[{tag}] sum = {sum(len(v) for v in fams.values())}")
    print(f"[{tag}] singletons members = {fams.get('singletons', [])}")
    return {"n_features": len(cols),
            "groups": {k: v for k, v in fams.items()},
            "sizes": {k: len(v) for k, v in fams.items()}}


def main():
    # utf-8-sig: the cache was written with encoding="utf-8-sig" (BOM present)
    df = pd.read_csv(DAY_CSV, nrows=200, encoding="utf-8-sig")
    print(f"raw columns = {len(df.columns)}  first = {df.columns[0]!r}")
    bom = pd.read_csv(DAY_CSV, nrows=5)
    print(f"without utf-8-sig, first column reads as {bom.columns[0]!r}")
    print(f"window_start_epoch dtype = {df['window_start_epoch'].dtype}")
    out = {
        "day_csv": str(DAY_CSV),
        "raw_columns": len(df.columns),
        "first_column_utf8sig": df.columns[0],
        "first_column_plain_utf8": bom.columns[0],
        "epoch_dtype": str(df["window_start_epoch"].dtype),
        "with_epoch_in_meta": report("META WITH epoch", df, META_WITH),
        "without_epoch_in_meta": report("META WITHOUT epoch", df, META_WITHOUT),
    }
    out_path = REPO / "results/two_channel_unsw_20260903/family_check.json"
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
