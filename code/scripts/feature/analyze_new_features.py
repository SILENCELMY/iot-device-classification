#!/usr/bin/env python3
"""
分析新特征（Burst + Direction）在 top-10 重要特征中的表现
"""
import argparse
import pandas as pd
from collections import Counter
from pathlib import Path

# 定义新特征（v2版本）
# 明确的新特征名称
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

# 新特征的前缀/模式（包括衍生特征）
NEW_FEATURE_PATTERNS = ['burst_', 'up_', 'down_', 'side_', 'other_', 'bssid_']

def is_new_feature(feat_name):
    """判断是否为新特征"""
    # 明确的新特征
    if feat_name in NEW_FEATURES_EXPLICIT:
        return True
    # 带有新特征前缀的衍生特征
    for pattern in NEW_FEATURE_PATTERNS:
        if feat_name.startswith(pattern):
            return True
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="分析新特征在 top-10 重要特征中的表现")
    parser.add_argument("--input", default="results/robust_v2/feature_rankings_all_tasks.csv")
    parser.add_argument("--output", default="results/robust_v2/report/new_features_top10_stats.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tasks = df['task'].unique()
    print("=" * 100)
    print("新特征（Burst + Direction）在各任务 Top-10 中的表现")
    print("=" * 100)

    top10_stats = []
    for task in tasks:
        task_df = df[df['task'] == task].sort_values('rank').head(10)
        new_feats = [f for f in task_df['feature'].values if is_new_feature(f)]

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
    print(f"总任务数: {len(tasks)}")
    print(f"平均每个任务 top-10 中的新特征数: {top10_df['new_feature_count'].mean():.2f}")
    print(f"最多的任务有 {top10_df['new_feature_count'].max()} 个新特征在 top-10")
    print(f"最少的任务有 {top10_df['new_feature_count'].min()} 个新特征在 top-10")

    all_top10_features = []
    for task in tasks:
        task_df = df[df['task'] == task].sort_values('rank').head(10)
        all_top10_features.extend(task_df['feature'].values)

    new_feat_counter = Counter([f for f in all_top10_features if is_new_feature(f)])

    print("\n" + "=" * 100)
    print("Top-10 中最常出现的新特征（跨所有任务）")
    print("=" * 100)
    for feat, count in new_feat_counter.most_common(20):
        print(f"{feat:30s} 出现 {count:2d} 次（在 {len(tasks)} 个任务中，占比 {count/len(tasks)*100:.1f}%）")

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
    print(f"新特征总计: {len(new_feat_counter)} 次")

    tasks_without_new = top10_df[top10_df['new_feature_count'] == 0]['task'].tolist()
    if tasks_without_new:
        print(f"\n⚠️ 以下任务的 top-10 中没有新特征: {', '.join(tasks_without_new)}")
    else:
        print(f"\n✅ 所有 {len(tasks)} 个任务的 top-10 中都有新特征出现！")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top10_df.to_csv(output_path, index=False)
    print(f"\n✅ 统计结果已保存: {args.output}")


if __name__ == "__main__":
    main()
