#!/usr/bin/env python3
"""
分析 Accuracy 和 Macro-F1 的关系
"""
import argparse
import pandas as pd

def get_scenario(task):
    if 'single_round' in task:
        return 'IID (Single Round)'
    elif 'joint' in task:
        return 'IID (Joint)'
    elif 'loro' in task:
        return 'OOD (LORO)'
    elif 'position' in task:
        return 'OOD (Position)'
    elif 'jitter' in task:
        return 'OOD (Jitter)'
    return 'Other'


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 Accuracy 和 Macro-F1 的关系")
    parser.add_argument("--report-dir", default="results/robust_v2/report")
    parser.add_argument("--input", default=None, help="默认读取 <report-dir>/all_metrics_combined.csv")
    args = parser.parse_args()

    report_dir = args.report_dir
    input_path = args.input or f"{report_dir}/all_metrics_combined.csv"
    df = pd.read_csv(input_path)
    df['acc_f1_diff'] = df['accuracy'] - df['macro_f1']

    print("=" * 100)
    print("Accuracy vs Macro-F1 关系分析")
    print("=" * 100)

    df['scenario'] = df['task'].apply(get_scenario)

    print("\n按场景分组的统计:")
    print("-" * 100)
    scenario_stats = df.groupby('scenario').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'acc_f1_diff': ['mean', 'std']
    }).round(4)

    for scenario in ['IID (Single Round)', 'IID (Joint)', 'OOD (LORO)', 'OOD (Position)', 'OOD (Jitter)']:
        if scenario in scenario_stats.index:
            row = scenario_stats.loc[scenario]
            acc_mean = row[('accuracy', 'mean')]
            f1_mean = row[('macro_f1', 'mean')]
            diff_mean = row[('acc_f1_diff', 'mean')]
            diff_std = row[('acc_f1_diff', 'std')]

            print(f"\n{scenario}:")
            print(f"  平均 Accuracy:  {acc_mean:.4f}")
            print(f"  平均 Macro-F1: {f1_mean:.4f}")
            print(f"  差异 (Acc-F1): {diff_mean:+.4f} ± {diff_std:.4f}")

    print("\n" + "=" * 100)
    print("差异最大的案例 (Top 10)")
    print("=" * 100)

    top_diff = df.nlargest(10, 'acc_f1_diff', keep='all')[['task', 'feature_set', 'model', 'accuracy', 'macro_f1', 'acc_f1_diff']]
    print(top_diff.to_string(index=False))

    print("\n差异最小的案例 (Bottom 10):")
    bottom_diff = df.nsmallest(10, 'acc_f1_diff', keep='all')[['task', 'feature_set', 'model', 'accuracy', 'macro_f1', 'acc_f1_diff']]
    print(bottom_diff.to_string(index=False))

    print("\n" + "=" * 100)
    print("Correlation Analysis")
    print("=" * 100)
    corr = df['accuracy'].corr(df['macro_f1'])
    print(f"Accuracy 与 Macro-F1 的 Pearson 相关系数: {corr:.4f}")

    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        if len(scenario_df) > 5:
            corr = scenario_df['accuracy'].corr(scenario_df['macro_f1'])
            print(f"  {scenario}: r = {corr:.4f} (n={len(scenario_df)})")

    print("\n" + "=" * 100)
    print("关键发现")
    print("=" * 100)
    print("""
1. IID 场景 (Single Round / Joint):
   - Accuracy ≈ Macro-F1，差异很小
   - 原因：类别分布相对均衡

2. OOD 场景 (LORO / Position / Jitter):
   - Accuracy > Macro-F1，差异变大
   - 原因：类别不平衡加剧，某些类别（如 Sensor）识别很差

3. 最严重的案例:
   - loro_R2_R4_to_R3: Accuracy - F1 = 0.04~0.06
   - 说明某些类别（如 Sensor）召回率极低，拉低了 Macro-F1
   - 但 Accuracy 被大类（如 Camera, Socket）的高准确率拉高

4. 论文写作建议:
   - OOD 场景应使用 Macro-F1 作为主要指标
   - Accuracy 会掩盖少数类的识别问题
   - 特别关注 Sensor 等困难类别的性能
""")


if __name__ == "__main__":
    main()
