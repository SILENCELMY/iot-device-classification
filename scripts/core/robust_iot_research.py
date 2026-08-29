import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


TSHARK_FIELDS = [
    "frame.time_epoch",
    "frame.len",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fc.retry",
    "wlan.fc.type_subtype",
    "radiotap.dbm_antsignal",
    "wlan.sa",
    "wlan.da",
    "wlan.bssid",
    "wlan.ta",
    "wlan.ra",
]
FILTER_EXPRESSIONS = {
    "raw_all": None,
    "data_only": "wlan.fc.type == 2",
    "data_non_null": "wlan.fc.type == 2 && !(wlan.fc.subtype == 4 || wlan.fc.subtype == 12)",
}
META_COLUMNS = {
    "label",
    "round",
    "traffic",
    "filter_mode",
    "source_file",
    "window_id",
    "window_start",
    "window_end",
}


def optional_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified robust IoT device recognition experiments."
    )
    parser.add_argument("--config", default="configs/research_experiments.json")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output-root", default="results/robust_iot_research")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--min-packets-per-window", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--filter-modes",
        default="raw_all",
        help="Comma-separated filter modes: raw_all,data_only,data_non_null.",
    )
    parser.add_argument(
        "--task-set",
        choices=["core", "filter", "all"],
        default="core",
        help="core excludes R1 filter-only task; filter focuses on R1/R2-R4; all runs every configured task.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task names. Overrides --task-set when provided.",
    )
    parser.add_argument(
        "--models",
        default="rf",
        help="Comma-separated models: rf,extra_trees,hist_gb,xgboost,lightgbm,stacking.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["all", "selected", "both"],
        default="both",
        help="Evaluate all features, selected features, or both.",
    )
    parser.add_argument("--disable-feature-selection", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Debug only: sample at most N rows per task after filtering by rounds.",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def required_rounds(tasks: list[dict[str, Any]]) -> set[str]:
    rounds: set[str] = set()
    for task in tasks:
        rounds.update(task.get("rounds", []))
        rounds.update(task.get("train_rounds", []))
        rounds.update(task.get("test_rounds", []))
    return rounds


def select_tasks(config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = config["evaluation_tasks"]
    if args.tasks:
        requested = set(comma_list(args.tasks))
        return [task for task in tasks if task["name"] in requested]
    if args.task_set == "all":
        return tasks
    if args.task_set == "filter":
        allowed_types = {"single_round", "joint_validation", "fixed_split"}
        return [
            task
            for task in tasks
            if task["type"] in allowed_types
            and not any(round_name in task.get("test_rounds", []) for round_name in ["R5", "R6", "R7"])
        ]
    return [task for task in tasks if task["name"] != "filtered_R1_single_round"]


def pcap_path_for(
    dataset_root: Path,
    device_dir: str,
    round_name: str,
    round_dir: str,
) -> Path:
    return dataset_root / device_dir / round_dir / f"{device_dir}_r{round_name[-1]}.pcapng"


def run_tshark(pcap_path: Path, filter_mode: str) -> pd.DataFrame:
    command = ["tshark", "-r", str(pcap_path)]
    display_filter = FILTER_EXPRESSIONS[filter_mode]
    if display_filter:
        command.extend(["-Y", display_filter])
    command.extend(["-T", "fields"])
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    command.extend(["-E", "header=n", "-E", "separator=\t", "-E", "occurrence=a"])

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < len(TSHARK_FIELDS):
            parts.extend([""] * (len(TSHARK_FIELDS) - len(parts)))
        rows.append(parts[: len(TSHARK_FIELDS)])
    return pd.DataFrame(
        rows,
        columns=[
            "time_epoch",
            "length",
            "fc_type",
            "fc_subtype",
            "retry",
            "type_subtype",
            "rssi",
            "sa",
            "da",
            "bssid",
            "ta",
            "ra",
        ],
    )


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_rssi(value: object) -> float:
    if not isinstance(value, str) or not value:
        return math.nan
    values = [to_float(item) for item in value.split(",")]
    values = [item for item in values if not math.isnan(item)]
    return float(np.mean(values)) if values else math.nan


def quantile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.quantile(q))


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def safe_std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def coefficient_of_variation(mean_value: float, std_value: float) -> float:
    return float(std_value / abs(mean_value)) if abs(mean_value) > 1e-12 else 0.0


def summarize_window(
    group: pd.DataFrame,
    label: str,
    round_name: str,
    traffic: str,
    filter_mode: str,
    source_file: Path,
    window_id: int,
    window_seconds: float,
) -> dict[str, Any]:
    lengths = group["length"].astype(float)
    times = group["time_epoch"].astype(float).sort_values()
    interarrival = times.diff().dropna()
    rssi = group["rssi_value"].dropna().astype(float)
    fc_type = group["fc_type"].astype("string")
    subtype = group["fc_subtype"].astype("string")
    retry = group["retry"].astype("string").str.lower()

    len_mean = safe_mean(lengths)
    len_std = safe_std(lengths)
    ia_mean = safe_mean(interarrival)
    ia_std = safe_std(interarrival)
    rssi_mean = safe_mean(rssi)
    rssi_std = safe_std(rssi)

    row: dict[str, Any] = {
        "label": label,
        "round": round_name,
        "traffic": traffic,
        "filter_mode": filter_mode,
        "source_file": str(source_file),
        "window_id": window_id,
        "window_start": float(group["relative_time"].min()),
        "window_end": float(group["relative_time"].max()),
        "packet_count": int(len(group)),
        "byte_count": float(lengths.sum()),
        "len_mean": len_mean,
        "len_std": len_std,
        "len_min": float(lengths.min()),
        "len_max": float(lengths.max()),
        "len_range": float(lengths.max() - lengths.min()),
        "len_cv": coefficient_of_variation(len_mean, len_std),
        "len_p10": quantile(lengths, 0.10),
        "len_p25": quantile(lengths, 0.25),
        "len_p50": quantile(lengths, 0.50),
        "len_p75": quantile(lengths, 0.75),
        "len_p90": quantile(lengths, 0.90),
        "len_p95": quantile(lengths, 0.95),
        "len_iqr": quantile(lengths, 0.75) - quantile(lengths, 0.25),
        "interarrival_mean": ia_mean,
        "interarrival_std": ia_std,
        "interarrival_min": float(interarrival.min()) if len(interarrival) else 0.0,
        "interarrival_max": float(interarrival.max()) if len(interarrival) else 0.0,
        "interarrival_cv": coefficient_of_variation(ia_mean, ia_std),
        "interarrival_p10": quantile(interarrival, 0.10),
        "interarrival_p25": quantile(interarrival, 0.25),
        "interarrival_p50": quantile(interarrival, 0.50),
        "interarrival_p75": quantile(interarrival, 0.75),
        "interarrival_p90": quantile(interarrival, 0.90),
        "interarrival_p95": quantile(interarrival, 0.95),
        "interarrival_iqr": quantile(interarrival, 0.75) - quantile(interarrival, 0.25),
        "data_ratio": float((fc_type == "2").mean()),
        "mgmt_ratio": float((fc_type == "0").mean()),
        "ctrl_ratio": float((fc_type == "1").mean()),
        "retry_ratio": float(retry.isin(["true", "1"]).mean()),
        "unique_sa_count": int(group["sa"].replace("", np.nan).nunique(dropna=True)),
        "unique_da_count": int(group["da"].replace("", np.nan).nunique(dropna=True)),
        "rssi_mean": rssi_mean,
        "rssi_std": rssi_std,
        "rssi_min": float(rssi.min()) if len(rssi) else 0.0,
        "rssi_max": float(rssi.max()) if len(rssi) else 0.0,
        "rssi_p10": quantile(rssi, 0.10),
        "rssi_p50": quantile(rssi, 0.50),
        "rssi_p90": quantile(rssi, 0.90),
        "rssi_missing_ratio": float(group["rssi_value"].isna().mean()),
    }

    for subtype_id in range(16):
        row[f"subtype_{subtype_id}_ratio"] = float((subtype == str(subtype_id)).mean())
    row["null_data_ratio"] = float(((fc_type == "2") & subtype.isin(["4", "12"])).mean())
    row["qos_data_ratio"] = float(((fc_type == "2") & (subtype == "8")).mean())

    burst_interarrival = interarrival <= 0.10
    row["burst_packet_ratio"] = float(burst_interarrival.mean()) if len(burst_interarrival) else 0.0
    row["long_gap_ratio"] = float((interarrival >= 1.0).mean()) if len(interarrival) else 0.0

    # --- Burst structure (v2): per-burst packet-count, duration, inter-burst gap ---
    burst_starts = np.concatenate([[True], ~burst_interarrival.to_numpy()[:-1]]) if len(burst_interarrival) else np.array([True])
    burst_id = np.cumsum(burst_starts.astype(int))
    burst_series = pd.Series(burst_id, index=interarrival.index)
    burst_sizes = burst_series.value_counts(sort=False)
    burst_count = int(len(burst_sizes))
    if burst_count > 0:
        burst_size_arr = burst_sizes.to_numpy()
        burst_size_mean = float(burst_size_arr.mean())
        burst_size_std = float(burst_size_arr.std(ddof=0)) if len(burst_size_arr) > 1 else 0.0
        burst_size_max = int(burst_size_arr.max())
        burst_size_min = int(burst_size_arr.min())
        burst_packet_fraction = float(burst_interarrival.sum()) / max(len(interarrival), 1)
    else:
        burst_size_mean = burst_size_std = burst_packet_fraction = 0.0
        burst_size_max = burst_size_min = 0
    row["burst_count"] = burst_count
    row["burst_size_mean"] = burst_size_mean
    row["burst_size_std"] = burst_size_std
    row["burst_size_max"] = burst_size_max
    row["burst_size_min"] = burst_size_min
    row["burst_packet_fraction"] = burst_packet_fraction

    # --- Direction inference (v2): classify each packet as up/down/other/other_ap ---
    # Heuristic: BSSID = the most frequent non-empty BSSID in the window (the AP).
    # In 802.11 terms, TA = Transmitter Address (the radio that actually sent the
    # frame on air), DA = Destination Address (the radio that receives it next on
    # air). This tshark 3.2.3 does not expose ToDS/FromDS, so we use TA/DA against
    # BSSID instead:
    #   TA == BSSID -> the AP transmitted the frame -> downlink (AP -> station)
    #   DA == BSSID -> the AP will receive the frame   -> uplink   (station -> AP)
    # Mgmt/Ctrl frames (TA-only broadcast, beacons) are labelled "side".
    bssid_series = group["bssid"].replace("", np.nan).dropna()
    bssid = bssid_series.mode().iloc[0] if not bssid_series.empty else ""
    ta = group["ta"].astype("string")
    da = group["da"].astype("string")
    if bssid:
        ap_is_tx = ta == bssid  # AP sent the frame -> downlink to a station
        ap_is_rx = da == bssid  # AP received the frame -> uplink from a station
    else:
        ap_is_tx = pd.Series([False] * len(group), index=group.index)
        ap_is_rx = pd.Series([False] * len(group), index=group.index)
    is_downlink = ap_is_tx & ~ap_is_rx
    is_uplink = ap_is_rx & ~ap_is_tx
    is_mgmt_or_ctrl = fc_type.isin(["0", "1"])
    direction_label = np.where(
        is_mgmt_or_ctrl, "side",
        np.where(is_downlink, "down", np.where(is_uplink, "up", "other"))
    )
    n = len(group)
    row["up_packet_ratio"] = float((direction_label == "up").sum()) / max(n, 1)
    row["down_packet_ratio"] = float((direction_label == "down").sum()) / max(n, 1)
    row["side_packet_ratio"] = float((direction_label == "side").sum()) / max(n, 1)
    row["other_packet_ratio"] = float((direction_label == "other").sum()) / max(n, 1)
    row["up_down_ratio"] = (
        float((direction_label == "up").sum()) / max(float((direction_label == "down").sum()), 1.0)
    )
    row["bssid_known"] = float(1.0 if bssid else 0.0)

    # --- Direction packet stats (v2): len/ia stats split by direction ---
    # direction_label has length == len(group); interarrival has length == len(group) - 1.
    # Align masks with both indices.
    up_mask_full = pd.Series(direction_label == "up", index=group.index)
    down_mask_full = pd.Series(direction_label == "down", index=group.index)
    ia_index = interarrival.index
    up_ia_mask = up_mask_full.reindex(ia_index, fill_value=False)
    down_ia_mask = down_mask_full.reindex(ia_index, fill_value=False)
    up_len = lengths[up_mask_full]
    down_len = lengths[down_mask_full]
    up_ia = interarrival[up_ia_mask]
    down_ia = interarrival[down_ia_mask]
    row["up_len_mean"] = safe_mean(up_len)
    row["up_len_std"] = safe_std(up_len)
    row["up_len_p50"] = quantile(up_len, 0.50)
    row["down_len_mean"] = safe_mean(down_len)
    row["down_len_std"] = safe_std(down_len)
    row["down_len_p50"] = quantile(down_len, 0.50)
    row["up_ia_mean"] = safe_mean(up_ia)
    row["up_ia_std"] = safe_std(up_ia)
    row["down_ia_mean"] = safe_mean(down_ia)
    row["down_ia_std"] = safe_std(down_ia)
    row["len_up_down_diff"] = abs(safe_mean(up_len) - safe_mean(down_len))

    sub_count = 5
    bins = np.linspace(0.0, window_seconds, sub_count + 1)
    positions = np.clip(group["relative_time"].to_numpy() - window_id * window_seconds, 0, window_seconds)
    packet_counts, _ = np.histogram(positions, bins=bins)
    byte_sums = []
    for idx in range(sub_count):
        if idx == sub_count - 1:
            mask = (positions >= bins[idx]) & (positions <= bins[idx + 1])
        else:
            mask = (positions >= bins[idx]) & (positions < bins[idx + 1])
        byte_sums.append(float(lengths.to_numpy()[mask].sum()))
    packet_counts_series = pd.Series(packet_counts.astype(float))
    byte_sums_series = pd.Series(byte_sums)
    sub_packet_mean = safe_mean(packet_counts_series)
    sub_packet_std = safe_std(packet_counts_series)
    row.update(
        {
            "subwin_packet_mean": sub_packet_mean,
            "subwin_packet_std": sub_packet_std,
            "subwin_packet_min": float(packet_counts_series.min()),
            "subwin_packet_max": float(packet_counts_series.max()),
            "subwin_packet_cv": coefficient_of_variation(sub_packet_mean, sub_packet_std),
            "subwin_byte_mean": safe_mean(byte_sums_series),
            "subwin_byte_std": safe_std(byte_sums_series),
            "subwin_byte_min": float(byte_sums_series.min()),
            "subwin_byte_max": float(byte_sums_series.max()),
            "active_subwin_count": int((packet_counts_series > 0).sum()),
        }
    )
    return row


def extract_features_for_file(
    pcap_path: Path,
    label: str,
    round_name: str,
    traffic: str,
    filter_mode: str,
    window_seconds: float,
    min_packets_per_window: int,
) -> pd.DataFrame:
    packets = run_tshark(pcap_path, filter_mode)
    packets["time_epoch"] = pd.to_numeric(packets["time_epoch"], errors="coerce")
    packets["length"] = pd.to_numeric(packets["length"], errors="coerce")
    packets = packets.dropna(subset=["time_epoch", "length"]).copy()
    packets = packets[packets["length"] > 0].copy()
    if packets.empty:
        return pd.DataFrame()
    packets["rssi_value"] = packets["rssi"].map(parse_rssi)
    packets["relative_time"] = packets["time_epoch"] - packets["time_epoch"].min()
    packets["window_id"] = np.floor(packets["relative_time"] / window_seconds).astype(int)
    rows = [
        summarize_window(
            group,
            label,
            round_name,
            traffic,
            filter_mode,
            pcap_path,
            int(window_id),
            window_seconds,
        )
        for window_id, group in packets.groupby("window_id", sort=True)
        if len(group) >= min_packets_per_window
    ]
    return pd.DataFrame(rows)


def build_feature_table(
    config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    required: set[str],
    filter_mode: str,
    window_seconds: float,
    min_packets_per_window: int,
    force_extract: bool,
) -> pd.DataFrame:
    cache_path = output_dir / f"features_{filter_mode}_w{window_seconds:g}.csv"
    if cache_path.exists() and not force_extract:
        features = pd.read_csv(cache_path)
        missing = required - set(features["round"].unique())
        if not missing:
            return features
    else:
        features = pd.DataFrame()
        missing = set(required)

    frames = [features] if not features.empty else []
    for round_name in sorted(missing):
        round_info = config["rounds"][round_name]
        for label, device_dir in config["device_dirs"].items():
            pcap_path = pcap_path_for(dataset_root, device_dir, round_name, round_info["dir"])
            if not pcap_path.exists():
                raise FileNotFoundError(f"Missing input file: {pcap_path}")
            print(f"Extracting {filter_mode} {round_name} {label}: {pcap_path}", flush=True)
            frames.append(
                extract_features_for_file(
                    pcap_path=pcap_path,
                    label=label,
                    round_name=round_name,
                    traffic=round_info["traffic"],
                    filter_mode=filter_mode,
                    window_seconds=window_seconds,
                    min_packets_per_window=min_packets_per_window,
                )
            )

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return merged


def feature_columns(features: pd.DataFrame) -> list[str]:
    columns = []
    for column in features.columns:
        if column in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(features[column]):
            columns.append(column)
    return columns


def clean_x(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return data[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_label_encoder(labels: list[str]) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(labels)
    return encoder


class SimpleStackingClassifier:
    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        final_estimator: Any,
        cv: int = 5,
        random_state: int = 42,
        oof_mode: str = "grouped",
    ) -> None:
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.random_state = random_state
        # 协议 §9.1：oof_mode="grouped"（默认）= 按轮次分组 / 单轮按时间块；
        # "random" = 历史原实现（StratifiedKFold 随机分折），仅保留作 E1 的对照臂。
        if oof_mode not in {"grouped", "random"}:
            raise ValueError(f"oof_mode must be 'grouped' or 'random', got {oof_mode!r}")
        self.oof_mode = oof_mode

    def _splitter(self, x: pd.DataFrame, y: np.ndarray, train_round, window_start):
        """按协议 §9.1 生成 OOF 折叠索引。

        grouped 模式：
        - 多源轮次：GroupKFold，group = round；
        - 单轮：按 window_start 排序后的连续时间块，不打散相邻窗口；
        random 模式：原 StratifiedKFold(shuffle=True) 逻辑，逐字保留。
        """
        if self.oof_mode == "random":
            min_class_count = min(np.bincount(y.astype(int)))
            n_splits = max(2, min(self.cv, int(min_class_count)))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            yield from splitter.split(x, y)
            return

        if train_round is None and window_start is None:
            raise ValueError(
                "oof_mode='grouped' 需要提供 train_round（多轮）或 window_start（单轮），"
                "否则无法按协议 §9.1 分组；若确要复现历史行为请显式传 oof_mode='random'（E1 A 臂）。"
            )

        n = len(y)
        if train_round is not None and len(np.unique(train_round)) >= 2:
            n_splits = max(2, min(self.cv, int(len(np.unique(train_round)))))
            yield from GroupKFold(n_splits=n_splits).split(x, y, groups=train_round)
            return

        # 单轮（或未给 round）：按时间区间分块。注意 window_start 是各采集文件的
        # 相对时间戳（每个 source_file 从 0 起计、并发采集），因此不能全局排序切片，
        # 必须按共享时间区间分块：同一文件内相邻窗口同折，跨文件同时段对齐。
        if window_start is None:
            raise ValueError("单轮 grouped 模式必须提供 window_start（协议 §9.1：按时间块划分）")
        ws = np.asarray(window_start, dtype=float)
        n_blocks = max(2, min(self.cv, len(ws)))
        edges = np.quantile(ws, np.linspace(0, 1, n_blocks + 1)[1:-1])
        block_id = np.searchsorted(edges, ws, side="right")
        for i in range(n_blocks):
            val_idx = np.flatnonzero(block_id == i)
            if len(val_idx) == 0:
                continue
            train_idx = np.flatnonzero(block_id != i)
            yield train_idx, val_idx

    def fit(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        train_round=None,
        window_start=None,
    ) -> "SimpleStackingClassifier":
        self.classes_ = np.unique(y)
        meta_parts = []
        self.named_estimators_ = {}

        for name, estimator in self.estimators:
            oof = np.zeros((len(x), len(self.classes_)))
            for train_idx, val_idx in self._splitter(x, y, train_round, window_start):
                fold_model = clone(estimator)
                fold_model.fit(x.iloc[train_idx], y[train_idx])
                proba = fold_model.predict_proba(x.iloc[val_idx])
                # 折内训练集可能缺类（G0 环境组合、UNSW 18 类不均衡）。predict_proba 的列
                # 对应 fold_model.classes_，必须映射回全局 self.classes_ 的列位，缺失类留 0；
                # 否则列数不符会 broadcast 报错，或在列序不同时静默错位。
                fold_classes = np.asarray(fold_model.classes_)
                col_idx = np.searchsorted(self.classes_, fold_classes)
                oof[np.ix_(np.asarray(val_idx), col_idx)] = proba
            fitted = clone(estimator)
            fitted.fit(x, y)
            self.named_estimators_[name] = fitted
            meta_parts.append(oof)

        meta_x = np.hstack(meta_parts)
        self.oof_meta_ = meta_x  # 协议 §20.2：存为 self.oof_meta_
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(meta_x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        meta_parts = [
            estimator.predict_proba(x)
            for estimator in self.named_estimators_.values()
        ]
        return self.final_estimator_.predict_proba(np.hstack(meta_parts))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.final_estimator_.predict(np.hstack([
            estimator.predict_proba(x)
            for estimator in self.named_estimators_.values()
        ]))


def build_model(name: str, random_state: int, n_jobs: int, class_count: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=random_state)
    if name == "xgboost":
        if not optional_module("xgboost"):
            return None
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "lightgbm":
        if not optional_module("lightgbm"):
            return None
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            objective="multiclass",
            num_class=class_count,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if name == "stacking":
        estimators = []
        rf = build_model("rf", random_state, n_jobs, class_count)
        estimators.append(("rf", rf))
        xgb = build_model("xgboost", random_state, n_jobs, class_count)
        lgbm = build_model("lightgbm", random_state, n_jobs, class_count)
        if xgb is not None:
            estimators.append(("xgboost", xgb))
        if lgbm is not None:
            estimators.append(("lightgbm", lgbm))
        if len(estimators) == 1:
            estimators.append(("extra_trees", build_model("extra_trees", random_state, n_jobs, class_count)))
        return SimpleStackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            cv=5,
            random_state=random_state,
            oof_mode="grouped",  # 协议 §9.1；E1 A 臂显式传 oof_mode="random"
        )
    raise ValueError(f"Unsupported model: {name}")


def model_feature_importance(model: Any, columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return (
            pd.DataFrame({"feature": columns, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    if hasattr(model, "estimators_"):
        rows = []
        for estimator_name, estimator in getattr(model, "named_estimators_", {}).items():
            if hasattr(estimator, "feature_importances_"):
                for feature, importance in zip(columns, estimator.feature_importances_):
                    rows.append(
                        {
                            "feature": feature,
                            "importance": float(importance),
                            "source": estimator_name,
                        }
                    )
        if rows:
            return (
                pd.DataFrame(rows)
                .groupby("feature", as_index=False)["importance"]
                .mean()
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
    return pd.DataFrame({"feature": columns, "importance": np.nan})


def remove_correlated_features(
    x: pd.DataFrame,
    y_encoded: np.ndarray,
    columns: list[str],
    threshold: float,
    random_state: int,
) -> pd.DataFrame:
    mi = mutual_info_classif(x[columns], y_encoded, random_state=random_state)
    ranking = (
        pd.DataFrame({"feature": columns, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )
    corr = x[columns].corr().abs().fillna(0.0)
    kept: list[str] = []
    dropped: list[str] = []
    for feature in ranking["feature"]:
        if all(corr.loc[feature, selected] < threshold for selected in kept):
            kept.append(feature)
        else:
            dropped.append(feature)
    ranking["correlation_kept"] = ranking["feature"].isin(kept)
    ranking["correlation_dropped"] = ranking["feature"].isin(dropped)
    return ranking


def shap_importance_if_available(
    model: Any,
    x_sample: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if not optional_module("shap"):
        return pd.DataFrame({"feature": columns, "shap_importance": np.nan})
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
        if isinstance(values, list):
            array = np.mean([np.abs(item).mean(axis=0) for item in values], axis=0)
        else:
            raw = np.asarray(values)
            if raw.ndim == 3:
                array = np.abs(raw).mean(axis=(0, 2))
            else:
                array = np.abs(raw).mean(axis=0)
        return pd.DataFrame({"feature": columns, "shap_importance": array})
    except Exception as exc:
        return pd.DataFrame(
            {
                "feature": columns,
                "shap_importance": np.nan,
                "shap_error": str(exc),
            }
        )


def normalize_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    max_value = numeric.max()
    if max_value <= 0:
        return numeric
    return numeric / max_value


def select_features(
    x_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[str], pd.DataFrame]:
    settings = config["feature_selection"]
    corr_threshold = float(settings["correlation_threshold"])
    all_columns = list(x_train.columns)
    corr_ranking = remove_correlated_features(
        x_train,
        y_train_encoded,
        all_columns,
        corr_threshold,
        args.random_state,
    )
    kept_columns = corr_ranking[corr_ranking["correlation_kept"]]["feature"].tolist()

    selector_model = ExtraTreesClassifier(
        n_estimators=400,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=args.n_jobs,
    )
    selector_model.fit(x_train[kept_columns], y_train_encoded)
    model_importance = model_feature_importance(selector_model, kept_columns).rename(
        columns={"importance": "model_importance"}
    )
    sample_size = min(len(x_train), 1000)
    sample = x_train[kept_columns].sample(sample_size, random_state=args.random_state)
    shap_importance = shap_importance_if_available(selector_model, sample, kept_columns)
    ranking = (
        corr_ranking.merge(model_importance, on="feature", how="left")
        .merge(shap_importance, on="feature", how="left")
    )
    ranking["mi_norm"] = normalize_score(ranking["mutual_info"])
    ranking["model_norm"] = normalize_score(ranking["model_importance"])
    ranking["shap_norm"] = normalize_score(ranking["shap_importance"])
    ranking["joint_score"] = ranking[["mi_norm", "model_norm", "shap_norm"]].mean(axis=1)
    ranking.loc[~ranking["correlation_kept"], "joint_score"] = -1.0
    ranking = ranking.sort_values("joint_score", ascending=False).reset_index(drop=True)

    candidate_ks = [int(k) for k in settings["candidate_top_k"]]
    candidate_ks = sorted({min(k, len(kept_columns)) for k in candidate_ks if k > 0})
    if not candidate_ks:
        candidate_ks = [len(kept_columns)]

    x_inner_train, x_inner_val, y_inner_train, y_inner_val = train_test_split(
        x_train,
        y_train_encoded,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_train_encoded,
    )
    best_k = candidate_ks[0]
    best_f1 = -1.0
    validation_rows = []
    for k in candidate_ks:
        cols = ranking.head(k)["feature"].tolist()
        model = RandomForestClassifier(
            n_estimators=250,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=args.n_jobs,
        )
        model.fit(x_inner_train[cols], y_inner_train)
        pred = model.predict(x_inner_val[cols])
        _, _, f1, _ = precision_recall_fscore_support(
            y_inner_val,
            pred,
            labels=np.arange(len(labels)),
            average="macro",
            zero_division=0,
        )
        validation_rows.append({"top_k": k, "macro_f1": float(f1)})
        if f1 > best_f1:
            best_f1 = float(f1)
            best_k = k

    ranking["selected"] = False
    ranking.loc[: best_k - 1, "selected"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "feature_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "feature_selection_topk_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return ranking.head(best_k)["feature"].tolist(), ranking


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> tuple[dict[str, Any], np.ndarray, dict]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            label: float(value) for label, value in zip(labels, per_f1)
        },
        "per_class_support": {
            label: int(value) for label, value in zip(labels, support)
        },
        "per_class_precision": {
            label: float(value) for label, value in zip(labels, per_precision)
        },
        "per_class_recall": {
            label: float(value) for label, value in zip(labels, per_recall)
        },
    }
    return metrics, cm, report


def sample_balanced(data: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if not max_rows or len(data) <= max_rows:
        return data.copy()
    label_count = data["label"].nunique()
    per_label = max(1, max_rows // label_count)
    samples = []
    for _, group in data.groupby("label", sort=False):
        samples.append(group.sample(min(len(group), per_label), random_state=random_state))
    return pd.concat(samples, ignore_index=True)


def task_data(
    features: pd.DataFrame,
    task: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    if task["type"] in {"single_round", "joint_validation"}:
        rounds = task["rounds"]
        data = features[features["round"].isin(rounds)].copy()
        data = sample_balanced(data, args.max_rows, args.random_state)
        meta = data[["round", "traffic", "filter_mode", "source_file", "window_id", "window_start", "window_end"]]
        train_idx, test_idx = train_test_split(
            data.index,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=data["label"],
        )
        return (
            data.loc[train_idx],
            data.loc[test_idx],
            data.loc[train_idx, "label"],
            data.loc[test_idx, "label"],
            meta.loc[train_idx],
            meta.loc[test_idx],
        )

    train_data = features[features["round"].isin(task["train_rounds"])].copy()
    test_data = features[features["round"].isin(task["test_rounds"])].copy()
    train_data = sample_balanced(train_data, args.max_rows, args.random_state)
    test_data = sample_balanced(test_data, args.max_rows, args.random_state)
    return (
        train_data,
        test_data,
        train_data["label"],
        test_data["label"],
        train_data[["round", "traffic", "filter_mode", "source_file", "window_id", "window_start", "window_end"]],
        test_data[["round", "traffic", "filter_mode", "source_file", "window_id", "window_start", "window_end"]],
    )


def evaluate_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    x_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    y_test_labels: pd.Series,
    meta_test: pd.DataFrame,
    selected_columns: list[str],
    labels: list[str],
    encoder: LabelEncoder,
    output_dir: Path,
    summary_base: dict[str, Any],
    meta_train: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 协议 §20.2：stacking 需要轮次/时间信息做分组 OOF（§9.1）
    fit_kwargs: dict[str, Any] = {}
    if isinstance(model, SimpleStackingClassifier):
        if meta_train is not None:
            if "round" in meta_train.columns:
                fit_kwargs["train_round"] = meta_train["round"].to_numpy()
            if "window_start" in meta_train.columns:
                fit_kwargs["window_start"] = meta_train["window_start"].to_numpy()
    model.fit(x_train[selected_columns], y_train_encoded, **fit_kwargs)
    pred_encoded = model.predict(x_test[selected_columns])
    pred_labels = encoder.inverse_transform(pred_encoded.astype(int))

    metrics, cm, report = metric_summary(
        y_test_labels.to_numpy(),
        pred_labels,
        labels,
    )
    predictions = meta_test.copy()
    predictions["true_label"] = y_test_labels.to_numpy()
    predictions["predicted_label"] = pred_labels
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    # 协议 §20.2：落盘 pred_proba.csv（概率行和为 1）与 oof_meta.csv（仅 stacking）
    # 列顺序以 model.classes_ 为准并回填缺失类为 0.0——G0 的部分环境组合可能缺类，
    # 直接按 encoder.classes_ 枚举会错位或越界。
    proba = model.predict_proba(x_test[selected_columns])
    model_classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1]))).astype(int)
    proba_full = np.zeros((len(proba), len(encoder.classes_)), dtype=float)
    proba_full[:, model_classes] = proba
    proba_df = meta_test[["round", "window_id", "window_start"]].copy()
    if "source_file" in meta_test.columns:
        proba_df.insert(0, "source_file", meta_test["source_file"].to_numpy())
    for i, cls in enumerate(encoder.classes_):
        proba_df[f"proba_{cls}"] = proba_full[:, i]
    proba_df["true_label"] = y_test_labels.to_numpy()
    proba_df.to_csv(output_dir / "pred_proba.csv", index=False, encoding="utf-8-sig")
    if isinstance(model, SimpleStackingClassifier) and getattr(model, "oof_meta_", None) is not None:
        oof = model.oof_meta_
        oof_class_names = encoder.inverse_transform(np.asarray(model.classes_).astype(int))
        oof_df = pd.DataFrame(
            oof,
            columns=[
                f"oof_{name}_{cls}"
                for name in model.named_estimators_
                for cls in oof_class_names
            ],
        )
        if meta_train is not None:
            for col in ("round", "window_id", "window_start"):
                if col in meta_train.columns:
                    oof_df.insert(0, col, meta_train[col].to_numpy())
        oof_df["true_label"] = encoder.inverse_transform(y_train_encoded.astype(int))
        oof_df.to_csv(output_dir / "oof_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    model_feature_importance(model, selected_columns).to_csv(
        output_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(selected_columns, f, indent=2, ensure_ascii=False)
    joblib.dump(model, output_dir / "model.joblib")

    summary = {
        **summary_base,
        "model": model_name,
        "feature_count": len(selected_columns),
        **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": cm.tolist(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def evaluate_task(
    features: pd.DataFrame,
    task: dict[str, Any],
    filter_mode: str,
    model_names: list[str],
    args: argparse.Namespace,
    config: dict[str, Any],
    output_root: Path,
    labels: list[str],
    all_summaries: list[dict[str, Any]],
    ranking_collector: list[pd.DataFrame],
) -> None:
    train_data, test_data, y_train, y_test, meta_train, meta_test = task_data(features, task, args)
    if train_data.empty or test_data.empty:
        print(f"Skipping {task['name']} for {filter_mode}: empty train/test data", flush=True)
        return

    columns = feature_columns(train_data)
    x_train = clean_x(train_data, columns)
    x_test = clean_x(test_data, columns)
    encoder = fit_label_encoder(labels)
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    task_dir = output_root / filter_mode / task["name"]
    feature_sets = {"all_features": columns}
    if not args.disable_feature_selection and args.feature_mode in {"selected", "both"}:
        fs_dir = task_dir / "feature_selection"
        selected_columns, ranking = select_features(
            x_train,
            y_train_encoded,
            labels,
            args,
            config,
            fs_dir,
        )
        ranking["filter_mode"] = filter_mode
        ranking["task"] = task["name"]
        ranking_collector.append(ranking)
        feature_sets["selected_features"] = selected_columns

    if args.feature_mode == "all":
        feature_sets = {"all_features": columns}
    elif args.feature_mode == "selected":
        feature_sets = {k: v for k, v in feature_sets.items() if k == "selected_features"}

    summary_base = {
        "filter_mode": filter_mode,
        "task": task["name"],
        "task_type": task["type"],
        "train_rounds": task.get("train_rounds", task.get("rounds", [])),
        "test_rounds": task.get("test_rounds", task.get("rounds", [])),
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
    }

    for feature_set_name, selected_columns in feature_sets.items():
        for model_name in model_names:
            model = build_model(model_name, args.random_state, args.n_jobs, len(labels))
            if model is None:
                print(f"Skipping unavailable model: {model_name}", flush=True)
                continue
            model_dir = task_dir / feature_set_name / model_name
            try:
                summary = evaluate_model(
                    model_name=model_name,
                    model=model,
                    x_train=x_train,
                    y_train_encoded=y_train_encoded,
                    x_test=x_test,
                    y_test_encoded=y_test_encoded,
                    y_test_labels=y_test,
                    meta_test=meta_test,
                    meta_train=meta_train,
                    selected_columns=selected_columns,
                    labels=labels,
                    encoder=encoder,
                    output_dir=model_dir,
                    summary_base={**summary_base, "feature_set": feature_set_name},
                )
                all_summaries.append(summary)
                print(
                    f"{filter_mode} {task['name']} {feature_set_name} {model_name}: "
                    f"macro_f1={summary['macro_f1']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    **summary_base,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "error": str(exc),
                }
                all_summaries.append(error)
                model_dir.mkdir(parents=True, exist_ok=True)
                with (model_dir / "error.json").open("w", encoding="utf-8") as f:
                    json.dump(error, f, indent=2, ensure_ascii=False)
                print(f"Error in {filter_mode} {task['name']} {model_name}: {exc}", flush=True)


def save_frame_stats(features: pd.DataFrame, output_dir: Path, filter_mode: str) -> None:
    ratio_columns = [
        "data_ratio",
        "mgmt_ratio",
        "ctrl_ratio",
        "null_data_ratio",
        "qos_data_ratio",
        "retry_ratio",
    ]
    rows = []
    for (round_name, label), group in features.groupby(["round", "label"]):
        row = {
            "filter_mode": filter_mode,
            "round": round_name,
            "label": label,
            "windows": int(len(group)),
            "packet_count_mean": float(group["packet_count"].mean()),
        }
        for column in ratio_columns:
            if column in group:
                row[column] = float(group[column].mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / f"frame_stats_{filter_mode}.csv", index=False, encoding="utf-8-sig")


def save_feature_stability(rankings: list[pd.DataFrame], output_dir: Path) -> None:
    if not rankings:
        return
    combined = pd.concat(rankings, ignore_index=True)
    combined["rank"] = combined.groupby(["filter_mode", "task"])["joint_score"].rank(
        method="first",
        ascending=False,
    )
    stability = (
        combined.groupby("feature")
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            selected_rate=("selected", "mean"),
            mean_joint_score=("joint_score", "mean"),
            mean_mutual_info=("mutual_info", "mean"),
            mean_model_importance=("model_importance", "mean"),
        )
        .reset_index()
        .sort_values(["selected_rate", "mean_joint_score"], ascending=[False, False])
    )
    stability.to_csv(output_dir / "feature_stability.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(output_dir / "feature_rankings_all_tasks.csv", index=False, encoding="utf-8-sig")


def save_summary(summaries: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    flat_rows = []
    for item in summaries:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"per_class_f1", "confusion_matrix"}
        }
        if "per_class_f1" in item:
            for label, value in item["per_class_f1"].items():
                row[f"f1_{label}"] = value
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(output_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")


def save_environment_report(output_root: Path, model_names: list[str]) -> None:
    report = {
        "available_optional_modules": {
            "xgboost": optional_module("xgboost"),
            "lightgbm": optional_module("lightgbm"),
            "shap": optional_module("shap"),
        },
        "requested_models": model_names,
    }
    with (output_root / "environment_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    labels = config["labels"]
    tasks = select_tasks(config, args)
    model_names = comma_list(args.models)
    filter_modes = comma_list(args.filter_modes)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    save_environment_report(output_root, model_names)

    all_summaries: list[dict[str, Any]] = []
    ranking_collector: list[pd.DataFrame] = []
    dataset_root = Path(args.dataset_root)

    for filter_mode in filter_modes:
        if filter_mode not in FILTER_EXPRESSIONS:
            raise ValueError(f"Unsupported filter mode: {filter_mode}")
        mode_tasks = tasks
        rounds = required_rounds(mode_tasks)
        mode_dir = output_root / filter_mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        features = build_feature_table(
            config=config,
            dataset_root=dataset_root,
            output_dir=mode_dir,
            required=rounds,
            filter_mode=filter_mode,
            window_seconds=args.window_seconds,
            min_packets_per_window=args.min_packets_per_window,
            force_extract=args.force_extract,
        )
        save_frame_stats(features, mode_dir, filter_mode)
        for task in mode_tasks:
            evaluate_task(
                features=features,
                task=task,
                filter_mode=filter_mode,
                model_names=model_names,
                args=args,
                config=config,
                output_root=output_root,
                labels=labels,
                all_summaries=all_summaries,
                ranking_collector=ranking_collector,
            )

    save_summary(all_summaries, output_root)
    save_feature_stability(ranking_collector, output_root)
    print(f"\nSaved robust IoT research outputs to: {output_root}")


if __name__ == "__main__":
    main()
