#!/usr/bin/env python3
"""
Confusion Pattern Drift Analysis
目标: 量化不同环境下 confusion structure 的差异
方法: Normalized confusion matrix + Frobenius distance + Cosine similarity
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import json


class ConfusionPatternAnalyzer:
    """分析跨环境的混淆模式漂移"""

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']
        self.n_classes = len(self.class_names)

    def load_confusion_matrix(self, task: str, model: str, feature_set: str = 'all_features') -> np.ndarray:
        """加载混淆矩阵"""
        cm_path = self.results_root / task / feature_set / model / 'confusion_matrix.csv'
        if not cm_path.exists():
            return None
        cm = pd.read_csv(cm_path, index_col=0)
        return cm.values

    def normalize_confusion_matrix(self, cm: np.ndarray, method: str = 'row') -> np.ndarray:
        """
        归一化混淆矩阵
        method: 'row' (按真实类别归一化), 'all' (全局归一化), 'none' (不归一化)
        """
        if method == 'row':
            # 按行归一化 (每个真实类别的分布)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 避免除零
            return cm / row_sums
        elif method == 'all':
            # 全局归一化
            total = cm.sum()
            return cm / total if total > 0 else cm
        else:
            return cm

    def frobenius_distance(self, cm1: np.ndarray, cm2: np.ndarray) -> float:
        """计算两个混淆矩阵的 Frobenius 距离"""
        return np.linalg.norm(cm1 - cm2, ord='fro')

    def cosine_similarity(self, cm1: np.ndarray, cm2: np.ndarray) -> float:
        """计算两个混淆矩阵的 cosine similarity (flatten 后)"""
        v1 = cm1.flatten()
        v2 = cm2.flatten()
        # cosine distance -> cosine similarity
        return 1 - cosine(v1, v2)

    def off_diagonal_pattern(self, cm: np.ndarray) -> np.ndarray:
        """提取混淆模式 (去掉对角线, 只看错误分布)"""
        cm_off = cm.copy()
        np.fill_diagonal(cm_off, 0)
        return cm_off

    def compute_pattern_similarity_matrix(
        self,
        tasks: list,
        model: str = 'rf',
        feature_set: str = 'all_features'
    ) -> tuple:
        """
        计算任务间的混淆模式相似度矩阵
        返回: (frobenius_dist_matrix, cosine_sim_matrix, confusion_matrices)
        """
        n = len(tasks)
        frobenius_matrix = np.zeros((n, n))
        cosine_matrix = np.zeros((n, n))
        confusion_matrices = {}

        # 加载所有混淆矩阵并归一化
        for task in tasks:
            cm = self.load_confusion_matrix(task, model, feature_set)
            if cm is None:
                print(f"Warning: {task}/{model} confusion matrix not found")
                confusion_matrices[task] = None
            else:
                # 按行归一化 (focus on error distribution per class)
                cm_norm = self.normalize_confusion_matrix(cm, method='row')
                confusion_matrices[task] = cm_norm

        # 计算成对距离/相似度
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                cm1 = confusion_matrices.get(task1)
                cm2 = confusion_matrices.get(task2)

                if cm1 is None or cm2 is None:
                    frobenius_matrix[i, j] = np.nan
                    cosine_matrix[i, j] = np.nan
                    continue

                # Frobenius distance (越小越相似)
                frobenius_matrix[i, j] = self.frobenius_distance(cm1, cm2)

                # Cosine similarity (越大越相似)
                cosine_matrix[i, j] = self.cosine_similarity(cm1, cm2)

        return frobenius_matrix, cosine_matrix, confusion_matrices

    def analyze_class_specific_drift(
        self,
        tasks: list,
        model: str = 'rf',
        feature_set: str = 'all_features'
    ) -> pd.DataFrame:
        """
        逐类分析混淆模式漂移
        返回: DataFrame with class-level similarity metrics
        """
        confusion_matrices = {}
        for task in tasks:
            cm = self.load_confusion_matrix(task, model, feature_set)
            if cm is not None:
                cm_norm = self.normalize_confusion_matrix(cm, method='row')
                confusion_matrices[task] = cm_norm

        results = []

        # 对每个类别，计算其在不同任务间的错误分布相似度
        for class_idx, class_name in enumerate(self.class_names):
            for i, task1 in enumerate(tasks):
                for j, task2 in enumerate(tasks):
                    if i >= j:  # 只看上三角
                        continue

                    cm1 = confusion_matrices.get(task1)
                    cm2 = confusion_matrices.get(task2)

                    if cm1 is None or cm2 is None:
                        continue

                    # 提取该类的错误分布向量 (该行, 去掉对角线)
                    error_dist1 = cm1[class_idx].copy()
                    error_dist2 = cm2[class_idx].copy()

                    # 去掉对角线 (正确分类)
                    error_dist1[class_idx] = 0
                    error_dist2[class_idx] = 0

                    # 重新归一化
                    sum1 = error_dist1.sum()
                    sum2 = error_dist2.sum()
                    if sum1 > 0:
                        error_dist1 /= sum1
                    if sum2 > 0:
                        error_dist2 /= sum2

                    # 计算相似度 (只有当至少一个有错误时)
                    if sum1 > 0 or sum2 > 0:
                        cos_sim = 1 - cosine(error_dist1, error_dist2) if sum1 > 0 and sum2 > 0 else 0
                        l1_dist = np.abs(error_dist1 - error_dist2).sum()
                    else:
                        cos_sim = 1.0  # 都没有错误, 完全一致
                        l1_dist = 0.0

                    results.append({
                        'class': class_name,
                        'task1': task1,
                        'task2': task2,
                        'cosine_similarity': cos_sim,
                        'l1_distance': l1_dist,
                        'error_rate_1': 1 - cm1[class_idx, class_idx],
                        'error_rate_2': 1 - cm2[class_idx, class_idx],
                    })

        return pd.DataFrame(results)

    def visualize_pattern_similarity(
        self,
        tasks: list,
        frobenius_matrix: np.ndarray,
        cosine_matrix: np.ndarray,
        output_path: str
    ):
        """可视化混淆模式相似度矩阵"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Frobenius distance (越小越相似, 用 reversed colormap)
        ax1 = axes[0]
        sns.heatmap(frobenius_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=tasks, yticklabels=tasks, ax=ax1,
                    cbar_kws={'label': 'Frobenius Distance'})
        ax1.set_title('Confusion Pattern Distance (Frobenius)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Task', fontsize=12)
        ax1.set_ylabel('Task', fontsize=12)

        # 2. Cosine similarity (越大越相似)
        ax2 = axes[1]
        sns.heatmap(cosine_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=tasks, yticklabels=tasks, ax=ax2,
                    vmin=0, vmax=1,
                    cbar_kws={'label': 'Cosine Similarity'})
        ax2.set_title('Confusion Pattern Similarity (Cosine)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Task', fontsize=12)
        ax2.set_ylabel('Task', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion pattern similarity matrix saved: {output_path}")

    def generate_report(
        self,
        tasks: list,
        model: str = 'rf',
        feature_set: str = 'all_features',
        output_dir: str = None
    ):
        """生成完整的混淆模式漂移分析报告"""
        if output_dir is None:
            output_dir = self.results_root.parent / 'report'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print("Confusion Pattern Drift Analysis")
        print("=" * 80)

        # 1. 计算成对相似度
        frobenius_matrix, cosine_matrix, confusion_matrices = self.compute_pattern_similarity_matrix(
            tasks, model, feature_set
        )

        # 2. 可视化
        self.visualize_pattern_similarity(
            tasks, frobenius_matrix, cosine_matrix,
            output_dir / f'confusion_pattern_similarity_{model}.png'
        )

        # 3. 逐类分析
        class_drift_df = self.analyze_class_specific_drift(tasks, model, feature_set)
        class_drift_df.to_csv(output_dir / f'class_level_confusion_drift_{model}.csv', index=False)
        print(f"✅ Class-level drift analysis saved: class_level_confusion_drift_{model}.csv")

        # 4. 生成 markdown 报告
        report_lines = []
        report_lines.append("# Confusion Pattern Drift Analysis\n")
        report_lines.append(f"**Model**: {model.upper()}  ")
        report_lines.append(f"**Feature Set**: {feature_set}  \n")

        report_lines.append("## 1. Pattern Similarity Matrix\n")
        report_lines.append(f"![Pattern Similarity](confusion_pattern_similarity_{model}.png)\n")

        report_lines.append("## 2. Key Findings\n")

        # 找到最相似和最不相似的任务对
        n = len(tasks)
        max_sim = -1
        min_sim = 2
        max_pair = None
        min_pair = None

        for i in range(n):
            for j in range(i+1, n):
                if not np.isnan(cosine_matrix[i, j]):
                    sim = cosine_matrix[i, j]
                    if sim > max_sim:
                        max_sim = sim
                        max_pair = (tasks[i], tasks[j])
                    if sim < min_sim:
                        min_sim = sim
                        min_pair = (tasks[i], tasks[j])

        report_lines.append(f"### Most Similar Confusion Patterns:\n")
        if max_pair:
            report_lines.append(f"- **{max_pair[0]} ↔ {max_pair[1]}**: cosine similarity = {max_sim:.3f}\n")

        report_lines.append(f"\n### Most Dissimilar Confusion Patterns:\n")
        if min_pair:
            report_lines.append(f"- **{min_pair[0]} ↔ {min_pair[1]}**: cosine similarity = {min_sim:.3f}\n")
            report_lines.append(f"  - Frobenius distance = {frobenius_matrix[tasks.index(min_pair[0]), tasks.index(min_pair[1])]:.3f}\n")

        # 逐类统计
        report_lines.append("\n## 3. Class-Level Drift Analysis\n")
        report_lines.append("| Class | Avg Cosine Sim | Std | Min | Max |\n")
        report_lines.append("|---|---|---|---|---|\n")

        for class_name in self.class_names:
            class_data = class_drift_df[class_drift_df['class'] == class_name]
            if len(class_data) > 0:
                avg_sim = class_data['cosine_similarity'].mean()
                std_sim = class_data['cosine_similarity'].std()
                min_sim = class_data['cosine_similarity'].min()
                max_sim = class_data['cosine_similarity'].max()
                report_lines.append(f"| {class_name} | {avg_sim:.3f} | {std_sim:.3f} | {min_sim:.3f} | {max_sim:.3f} |\n")

        # 保存报告
        report_path = output_dir / f'confusion_pattern_drift_report_{model}.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"✅ Report saved: {report_path}")

        return {
            'frobenius_matrix': frobenius_matrix,
            'cosine_matrix': cosine_matrix,
            'class_drift_df': class_drift_df,
            'confusion_matrices': confusion_matrices
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Confusion Pattern Drift Analysis')
    parser.add_argument('--results-root', type=str, required=True,
                        help='Root directory of results (e.g., results/robust_v2/raw_all)')
    parser.add_argument('--model', type=str, default='rf',
                        help='Model name (rf, lightgbm, xgboost, stacking)')
    parser.add_argument('--feature-set', type=str, default='all_features',
                        help='Feature set (all_features, selected_features)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for reports')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='List of tasks to analyze (default: all LORO + single-round)')

    args = parser.parse_args()

    # 默认任务列表: LORO + single-round (用于比较训练 vs 测试的混淆模式)
    if args.tasks is None:
        args.tasks = [
            'single_round_R2', 'single_round_R3', 'single_round_R4',
            'loro_R2_R3_to_R4', 'loro_R2_R4_to_R3', 'loro_R3_R4_to_R2',
            'joint_R2_R3_R4',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6', 'jitter_R2_R3_R4_to_R7'
        ]

    analyzer = ConfusionPatternAnalyzer(args.results_root)
    results = analyzer.generate_report(
        tasks=args.tasks,
        model=args.model,
        feature_set=args.feature_set,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
