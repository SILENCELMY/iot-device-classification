#!/usr/bin/env python3
"""
聚合所有11个任务的特征排名，分析新特征表现
"""
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

# 11个任务
TASKS = [
    'single_round_R2', 'single_round_R3', 'single_round_R4',
    'joint_R2_R3_R4',
    'loro_R2_R3_to_R4', 'loro_R2_R4_to_R3', 'loro_R3_R4_to_R2',
    'position_R2_R3_R4_to_R5',
    'jitter_R2_R3_R4_to_R6', 'jitter_R2_R3_R4_to_R7', 'jitter_R2_R3_R4_to_R6_R7',
]

# 定义新特征
NEW_FEATURES_EXPLICIT = {
    # Burst 特征（6个）
    'burst_count', 'burst_size_mean', 'burst_size_std',
    'burst_size_max', 'burst_size_min', 'burst_packet_fraction',
    # 方向特征（10个）
    'up_packet_ratio', 'down_packet_ratio', 'side_packet_ratio', 'other_packet_ratio',
    'up_down_ratio', 'bssid_known',
    'up_len_mean', 'down_len_mean',
    'up_ia_mean', 'down_ia_mean',
    # 方向差异（1个）
    'len_up_down_diff',
}

NEW_FEATURE_PATTERNS = ['burst_', 'up_', 'down_', 'side_', 'other_', 'bssid_']

def is_new_feature(feat_name):
    """判断是否为新特征"""
    if feat_name in NEW_FEATURES_EXPLICIT:
        return True
    for pattern in NEW_FEATURE_PATTERNS:
        if feat_name.startswith(pattern):
            return True
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="聚合所有任务的特征排名，分析新特征表现")
    parser.add_argument("--results-root", type=Path, default=Path("results/robust_v2"))
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()

    raw_root = args.raw_root or args.results_root / "raw_all"
    report_dir = args.results_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rankings = []
    for task in TASKS:
        ranking_file = raw_root / task / 'feature_selection' / 'feature_ranking.csv'
        if not ranking_file.exists():
            print(f"⚠️ 文件不存在: {task}")
            continue

        df = pd.read_csv(ranking_file)
        df['task'] = task
        all_rankings.append(df)
        print(f"✅ 已加载: {task} ({len(df)} 个特征)")

    if all_rankings:
        full_df = pd.concat(all_rankings, ignore_index=True)
        full_df.to_csv(args.results_root / 'feature_rankings_all_tasks_complete.csv', index=False)
        print(f"\n✅ 完整特征排名已保存: feature_rankings_all_tasks_complete.csv ({len(full_df)} 行)")
    else:
        print("❌ 没有找到任何特征排名文件")
        raise SystemExit(1)

    print("\n" + "=" * 100)
    print("新特征（Burst + Direction）在各任务 Top-10 中的表现")
    print("=" * 100)

    top10_stats = []
    all_top10_features = []

    for task in TASKS:
        task_df = full_df[full_df['task'] == task]
        if len(task_df) == 0:
            continue

        if 'rank' in task_df.columns:
            task_top10 = task_df.sort_values('rank').head(10)
        elif 'joint_score' in task_df.columns:
            task_top10 = task_df.sort_values('joint_score', ascending=False).head(10)
        else:
            print(f"⚠️ {task} 没有 rank 或 joint_score 列")
            continue

        new_feats = [f for f in task_top10['feature'].values if is_new_feature(f)]
        all_top10_features.extend(task_top10['feature'].values)

        top10_stats.append({
            'task': task,
            'new_feature_count': len(new_feats),
            'new_features': ', '.join(new_feats[:5])
        })

        print(f"\n{task}:")
        print(f"  Top-10 中新特征数量: {len(new_feats)}/10")
        if new_feats:
            print(f"  新特征: {', '.join(new_feats)}")

    top10_df = pd.DataFrame(top10_stats)
    print("\n" + "=" * 100)
    print("汇总统计")
    print("=" * 100)
    print(f"总任务数: {len(top10_stats)}")
    print(f"平均每个任务 top-10 中的新特征数: {top10_df['new_feature_count'].mean():.2f}")
    print(f"最多的任务有 {top10_df['new_feature_count'].max()} 个新特征在 top-10")
    print(f"最少的任务有 {top10_df['new_feature_count'].min()} 个新特征在 top-10")

    new_feat_counter = Counter([f for f in all_top10_features if is_new_feature(f)])

    print("\n" + "=" * 100)
    print("Top-10 中最常出现的新特征（跨所有任务）")
    print("=" * 100)
    for feat, count in new_feat_counter.most_common(20):
        pct = count / len(top10_stats) * 100
        print(f"{feat:30s} 出现 {count:2d} 次（在 {len(top10_stats)} 个任务中，占比 {pct:5.1f}%）")

    burst_count = sum(1 for f in all_top10_features if f.startswith('burst_'))
    up_count = sum(1 for f in all_top10_features if f.startswith('up_'))
    down_count = sum(1 for f in all_top10_features if f.startswith('down_'))
    other_count = sum(1 for f in all_top10_features if f.startswith('other_') or f.startswith('side_'))

    print("\n" + "=" * 100)
    print("新特征类别统计（Top-10 中的总出现次数）")
    print("=" * 100)
    print(f"Burst 特征: {burst_count} 次")
    print(f"Up 方向特征: {up_count} 次")
    print(f"Down 方向特征: {down_count} 次")
    print(f"Other/Side 方向特征: {other_count} 次")
    print(f"新特征总出现次数: {len([f for f in all_top10_features if is_new_feature(f)])} 次")
    print(f"新特征种类数: {len(new_feat_counter)} 种")

    tasks_without_new = top10_df[top10_df['new_feature_count'] == 0]['task'].tolist()
    if tasks_without_new:
        print(f"\n⚠️ 以下任务的 top-10 中没有新特征:")
        for t in tasks_without_new:
            print(f"    - {t}")
    else:
        print(f"\n✅ 所有 {len(top10_stats)} 个任务的 top-10 中都有新特征出现！")

    top10_df.to_csv(report_dir / 'new_features_top10_stats_complete.csv', index=False)
    print(f"\n✅ 完整统计结果已保存: {report_dir / 'new_features_top10_stats_complete.csv'}")


if __name__ == "__main__":
    main()
