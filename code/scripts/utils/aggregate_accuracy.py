#!/usr/bin/env python3
"""
聚合所有任务的 accuracy 数据，并生成补充报告
"""
import json
import argparse
import pandas as pd
from pathlib import Path

# 11 个任务
TASKS = [
    'single_round_R2', 'single_round_R3', 'single_round_R4',
    'joint_R2_R3_R4',
    'loro_R2_R3_to_R4', 'loro_R2_R4_to_R3', 'loro_R3_R4_to_R2',
    'position_R2_R3_R4_to_R5',
    'jitter_R2_R3_R4_to_R6', 'jitter_R2_R3_R4_to_R7', 'jitter_R2_R3_R4_to_R6_R7',
]
FEATURE_SETS = ['all_features', 'selected_features']
MODELS = ['rf', 'xgboost', 'lightgbm', 'extra_trees', 'stacking']


def make_pivot(metric: str) -> pd.DataFrame:
    """生成 task × feature_set × model 的透视表"""
    pivot = df.pivot_table(
        index=['task', 'feature_set'],
        columns='model',
        values=metric,
    ).reset_index()
    # 重命名列
    pivot.columns.name = None
    return pivot


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合所有任务的 accuracy/precision/recall/F1 数据")
    parser.add_argument("--results-root", type=Path, default=Path("results/robust_v2"))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    raw_root = args.raw_root or args.results_root / "raw_all"
    output_dir = args.output_dir or args.results_root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in TASKS:
        for fs in FEATURE_SETS:
            for model in MODELS:
                mp = raw_root / task / fs / model / 'metrics.json'
                if not mp.exists():
                    continue
                with open(mp) as f:
                    m = json.load(f)
                rows.append({
                    'task': task,
                    'feature_set': fs,
                    'model': model,
                    'accuracy': m.get('accuracy'),
                    'precision': m.get('precision'),
                    'recall': m.get('recall'),
                    'macro_f1': m.get('macro_f1'),
                })

    global df
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'all_metrics_combined.csv', index=False)
    print(f"✅ 已保存: {output_dir / 'all_metrics_combined.csv'} ({len(df)} 行)")

    for metric in ['accuracy', 'precision', 'recall', 'macro_f1']:
        pivot = make_pivot(metric)
        pivot.to_csv(output_dir / f'pivot_{metric}.csv', index=False)
        print(f"✅ 已保存: pivot_{metric}.csv")

    acc_pivot = make_pivot('accuracy')
    print("\n" + "=" * 100)
    print("ACCURACY 表（任务 × 特征集 × 模型）")
    print("=" * 100)

    fs_label = {'all_features': '全量特征', 'selected_features': '筛选特征'}
    header = "| 评估任务场景 | 特征子集 | extra_trees | lightgbm | rf | stacking | xgboost |"
    sep = "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for _, row in acc_pivot.iterrows():
        line = f"| ('{row['task']}', '{row['feature_set']}') | {fs_label[row['feature_set']]} | "
        line += f"{row.get('extra_trees', 0):.4f} | {row.get('lightgbm', 0):.4f} | "
        line += f"{row.get('rf', 0):.4f} | {row.get('stacking', 0):.4f} | "
        line += f"{row.get('xgboost', 0):.4f} |"
        print(line)

    for metric, name in [('precision', 'Precision (宏平均)'), ('recall', 'Recall (宏平均)')]:
        pivot = make_pivot(metric)
        print(f"\n{name} 表：")
        print(header)
        print(sep)
        for _, row in pivot.iterrows():
            line = f"| ('{row['task']}', '{row['feature_set']}') | {fs_label[row['feature_set']]} | "
            line += f"{row.get('extra_trees', 0):.4f} | {row.get('lightgbm', 0):.4f} | "
            line += f"{row.get('rf', 0):.4f} | {row.get('stacking', 0):.4f} | "
            line += f"{row.get('xgboost', 0):.4f} |"
            print(line)


if __name__ == "__main__":
    df = pd.DataFrame()
    main()
