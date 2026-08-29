#!/usr/bin/env python3
"""
Error Correlation Stability Analysis
目标: 分析不同环境下 model error correlation 的结构性变化
方法: 计算 base models 在不同任务上的错误相关性，并比较跨环境的稳定性
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import json


class ErrorCorrelationAnalyzer:
    """分析跨环境的模型错误相关性稳定性"""

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']

    def load_predictions(self, task: str, model: str, feature_set: str = 'all_features') -> pd.DataFrame:
        """加载预测结果"""
        pred_path = self.results_root / task / feature_set / model / 'predictions.csv'
        if not pred_path.exists():
            return None
        return pd.read_csv(pred_path)

    def compute_error_vector(self, pred_df: pd.DataFrame) -> np.ndarray:
        """
        计算错误向量 (每个样本是否预测错误)
        返回: binary vector, 1 = error, 0 = correct
        """
        return (pred_df['predicted_label'] != pred_df['true_label']).astype(int).values

    def compute_pairwise_error_correlation(
        self,
        task: str,
        models: list,
        feature_set: str = 'all_features',
        method: str = 'pearson'
    ) -> np.ndarray:
        """
        计算 base models 在给定任务上的错误相关性矩阵
        method: 'pearson' (线性相关) 或 'jaccard' (集合重叠)
        """
        error_vectors = {}

        # 加载所有模型的错误向量
        for model in models:
            pred_df = self.load_predictions(task, model, feature_set)
            if pred_df is None:
                return None
            error_vectors[model] = self.compute_error_vector(pred_df)

        n = len(models)
        corr_matrix = np.zeros((n, n))

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                err1 = error_vectors[model1]
                err2 = error_vectors[model2]

                if method == 'pearson':
                    # Pearson correlation
                    if err1.std() == 0 or err2.std() == 0:
                        # 如果有模型没有错误或全错，相关性无定义
                        corr = 1.0 if np.array_equal(err1, err2) else 0.0
                    else:
                        corr, _ = pearsonr(err1, err2)
                    corr_matrix[i, j] = corr

                elif method == 'jaccard':
                    # Jaccard similarity (intersection over union of error sets)
                    intersection = np.sum((err1 == 1) & (err2 == 1))
                    union = np.sum((err1 == 1) | (err2 == 1))
                    jaccard = intersection / union if union > 0 else 0.0
                    corr_matrix[i, j] = jaccard

                elif method == 'overlap':
                    # Error overlap ratio (共同错误 / 总错误)
                    total_errors = np.sum(err1) + np.sum(err2)
                    common_errors = 2 * np.sum((err1 == 1) & (err2 == 1))
                    overlap = common_errors / total_errors if total_errors > 0 else 0.0
                    corr_matrix[i, j] = overlap

        return corr_matrix

    def compute_correlation_stability(
        self,
        tasks: list,
        models: list,
        feature_set: str = 'all_features',
        method: str = 'pearson'
    ) -> dict:
        """
        计算跨任务的错误相关性稳定性
        返回: {
            'correlation_matrices': {task: corr_matrix},
            'stability_matrix': pairwise stability between tasks,
            'mean_correlation': 各任务的平均相关性
        }
        """
        correlation_matrices = {}
        mean_correlations = {}

        # 1. 计算每个任务的错误相关性矩阵
        for task in tasks:
            corr_matrix = self.compute_pairwise_error_correlation(task, models, feature_set, method)
            if corr_matrix is not None:
                correlation_matrices[task] = corr_matrix

                # 计算平均相关性 (去掉对角线)
                mask = ~np.eye(len(models), dtype=bool)
                mean_correlations[task] = corr_matrix[mask].mean()

        # 2. 计算任务间的相关性结构稳定性
        n_tasks = len(correlation_matrices)
        task_names = list(correlation_matrices.keys())
        stability_matrix = np.zeros((n_tasks, n_tasks))

        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                corr1 = correlation_matrices[task1]
                corr2 = correlation_matrices[task2]

                # 提取上三角 (去掉对角线)
                mask = np.triu(np.ones_like(corr1, dtype=bool), k=1)
                vec1 = corr1[mask]
                vec2 = corr2[mask]

                # 计算两个相关性向量的相关性 (meta-correlation)
                if vec1.std() > 0 and vec2.std() > 0:
                    stability, _ = pearsonr(vec1, vec2)
                else:
                    stability = 1.0 if np.allclose(vec1, vec2) else 0.0

                stability_matrix[i, j] = stability

        return {
            'correlation_matrices': correlation_matrices,
            'stability_matrix': stability_matrix,
            'mean_correlations': mean_correlations,
            'task_names': task_names
        }

    def analyze_class_specific_error_correlation(
        self,
        task: str,
        models: list,
        feature_set: str = 'all_features'
    ) -> dict:
        """
        逐类分析错误相关性
        返回: 每个类别上 models 的错误相关性
        """
        class_correlations = {}

        # 加载所有模型的预测
        predictions = {}
        for model in models:
            pred_df = self.load_predictions(task, model, feature_set)
            if pred_df is None:
                return None
            predictions[model] = pred_df

        # 对每个类别
        for class_name in self.class_names:
            # 筛选该类的样本
            class_mask = predictions[models[0]]['true_label'] == class_name
            n_samples = class_mask.sum()

            if n_samples == 0:
                continue

            # 计算该类上的错误向量
            error_vectors = {}
            for model in models:
                pred_df = predictions[model]
                errors = (pred_df['predicted_label'] != pred_df['true_label']).values
                error_vectors[model] = errors[class_mask]

            # 计算相关性矩阵
            n = len(models)
            corr_matrix = np.zeros((n, n))

            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    err1 = error_vectors[model1]
                    err2 = error_vectors[model2]

                    # Jaccard (更适合小样本)
                    intersection = np.sum((err1 == 1) & (err2 == 1))
                    union = np.sum((err1 == 1) | (err2 == 1))
                    jaccard = intersection / union if union > 0 else 0.0
                    corr_matrix[i, j] = jaccard

            class_correlations[class_name] = {
                'correlation_matrix': corr_matrix,
                'n_samples': n_samples,
                'error_rates': {model: error_vectors[model].mean() for model in models}
            }

        return class_correlations

    def visualize_error_correlation_stability(
        self,
        stability_results: dict,
        models: list,
        output_path: str
    ):
        """可视化错误相关性稳定性"""
        correlation_matrices = stability_results['correlation_matrices']
        stability_matrix = stability_results['stability_matrix']
        task_names = stability_results['task_names']

        n_tasks = len(task_names)
        n_cols = 3
        n_rows = (n_tasks + n_cols - 1) // n_cols

        # 1. 每个任务的错误相关性矩阵
        fig1, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        axes = axes.flatten() if n_tasks > 1 else [axes]

        for idx, task in enumerate(task_names):
            ax = axes[idx]
            corr_matrix = correlation_matrices[task]

            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       xticklabels=models, yticklabels=models, ax=ax,
                       vmin=-1, vmax=1, center=0,
                       cbar_kws={'label': 'Error Correlation'})
            ax.set_title(f'{task}', fontsize=11, fontweight='bold')

        # 隐藏多余的子图
        for idx in range(n_tasks, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig1.savefig(output_path.replace('.png', '_per_task.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Per-task error correlation saved: {output_path.replace('.png', '_per_task.png')}")

        # 2. 跨任务稳定性矩阵
        fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

        sns.heatmap(stability_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   xticklabels=task_names, yticklabels=task_names, ax=ax,
                   vmin=-1, vmax=1, center=0,
                   cbar_kws={'label': 'Correlation Structure Similarity'})
        ax.set_title('Error Correlation Stability Across Tasks', fontsize=14, fontweight='bold')
        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel('Task', fontsize=12)

        plt.tight_layout()
        fig2.savefig(output_path.replace('.png', '_stability.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Correlation stability matrix saved: {output_path.replace('.png', '_stability.png')}")

        # 3. 平均相关性趋势
        fig3, ax = plt.subplots(1, 1, figsize=(12, 6))

        mean_correlations = stability_results['mean_correlations']
        tasks_sorted = sorted(mean_correlations.keys(), key=lambda x: mean_correlations[x])
        values = [mean_correlations[t] for t in tasks_sorted]

        colors = ['red' if 'loro' in t.lower() else 'orange' if 'position' in t.lower()
                  else 'green' if 'jitter' in t.lower() else 'blue' if 'joint' in t.lower()
                  else 'gray' for t in tasks_sorted]

        ax.barh(range(len(tasks_sorted)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(tasks_sorted)))
        ax.set_yticklabels(tasks_sorted, fontsize=10)
        ax.set_xlabel('Mean Error Correlation', fontsize=12)
        ax.set_title('Mean Error Correlation Across Tasks', fontsize=14, fontweight='bold')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Moderate correlation')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig3.savefig(output_path.replace('.png', '_mean_corr.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Mean correlation plot saved: {output_path.replace('.png', '_mean_corr.png')}")

    def generate_report(
        self,
        tasks: list,
        models: list = None,
        feature_set: str = 'all_features',
        output_dir: str = None
    ):
        """生成完整的错误相关性稳定性分析报告"""
        if models is None:
            models = ['rf', 'xgboost', 'lightgbm']

        if output_dir is None:
            output_dir = self.results_root.parent / 'report'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print("Error Correlation Stability Analysis")
        print("=" * 80)

        # 1. 计算跨任务稳定性
        stability_results = self.compute_correlation_stability(tasks, models, feature_set)

        # 2. 可视化
        self.visualize_error_correlation_stability(
            stability_results, models,
            str(output_dir / 'error_correlation_stability.png')
        )

        # 3. 生成统计表
        mean_correlations = stability_results['mean_correlations']
        stability_matrix = stability_results['stability_matrix']
        task_names = stability_results['task_names']

        stats_df = pd.DataFrame({
            'task': list(mean_correlations.keys()),
            'mean_error_correlation': list(mean_correlations.values())
        })
        stats_df = stats_df.sort_values('mean_error_correlation', ascending=False)
        stats_df.to_csv(output_dir / 'error_correlation_stats.csv', index=False)
        print(f"✅ Error correlation stats saved: error_correlation_stats.csv")

        # 4. 生成 markdown 报告
        report_lines = []
        report_lines.append("# Error Correlation Stability Analysis\n")
        report_lines.append(f"**Base Models**: {', '.join([m.upper() for m in models])}  ")
        report_lines.append(f"**Feature Set**: {feature_set}  \n")

        report_lines.append("## 1. Error Correlation Matrices (Per Task)\n")
        report_lines.append("![Per-Task Correlation](error_correlation_stability_per_task.png)\n")

        report_lines.append("## 2. Correlation Structure Stability\n")
        report_lines.append("![Stability Matrix](error_correlation_stability_stability.png)\n")

        report_lines.append("## 3. Mean Error Correlation Across Tasks\n")
        report_lines.append("![Mean Correlation](error_correlation_stability_mean_corr.png)\n")

        report_lines.append("## 4. Key Findings\n")

        # 找到最稳定和最不稳定的任务对
        n = len(task_names)
        max_stab = -1
        min_stab = 2
        max_pair = None
        min_pair = None

        for i in range(n):
            for j in range(i+1, n):
                stab = stability_matrix[i, j]
                if stab > max_stab:
                    max_stab = stab
                    max_pair = (task_names[i], task_names[j])
                if stab < min_stab:
                    min_stab = stab
                    min_pair = (task_names[i], task_names[j])

        report_lines.append(f"### Most Stable Error Correlation Structure:\n")
        if max_pair:
            report_lines.append(f"- **{max_pair[0]} ↔ {max_pair[1]}**: meta-correlation = {max_stab:.3f}\n")
            report_lines.append(f"  - Interpretation: base models make **consistent errors** across these two environments\n")

        report_lines.append(f"\n### Most Unstable Error Correlation Structure:\n")
        if min_pair:
            report_lines.append(f"- **{min_pair[0]} ↔ {min_pair[1]}**: meta-correlation = {min_stab:.3f}\n")
            report_lines.append(f"  - Interpretation: base models make **different types of errors** in these two environments\n")

        # 高相关性 vs 低相关性任务
        high_corr_tasks = [t for t, v in mean_correlations.items() if v > 0.5]
        low_corr_tasks = [t for t, v in mean_correlations.items() if v < 0.3]

        report_lines.append(f"\n### High Error Correlation Tasks (> 0.5):\n")
        if high_corr_tasks:
            for task in high_corr_tasks:
                report_lines.append(f"- {task}: {mean_correlations[task]:.3f}\n")
            report_lines.append(f"\n**Implication**: Base models are making **similar errors** → Stacking has limited diversity benefit\n")

        report_lines.append(f"\n### Low Error Correlation Tasks (< 0.3):\n")
        if low_corr_tasks:
            for task in low_corr_tasks:
                report_lines.append(f"- {task}: {mean_correlations[task]:.3f}\n")
            report_lines.append(f"\n**Implication**: Base models are making **diverse errors** → Stacking could potentially benefit (if errors are random, not systematic)\n")

        report_lines.append("\n## 5. Statistical Summary\n")
        report_lines.append("| Task | Mean Error Correlation |\n")
        report_lines.append("|---|---|\n")
        for _, row in stats_df.iterrows():
            report_lines.append(f"| {row['task']} | {row['mean_error_correlation']:.3f} |\n")

        # 保存报告
        report_path = output_dir / 'error_correlation_stability_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"✅ Report saved: {report_path}")

        return stability_results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Error Correlation Stability Analysis')
    parser.add_argument('--results-root', type=str, required=True,
                        help='Root directory of results')
    parser.add_argument('--models', type=str, nargs='+', default=['rf', 'xgboost', 'lightgbm'],
                        help='Base models to analyze')
    parser.add_argument('--feature-set', type=str, default='all_features',
                        help='Feature set')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='List of tasks')

    args = parser.parse_args()

    if args.tasks is None:
        args.tasks = [
            'single_round_R2', 'single_round_R3', 'single_round_R4',
            'loro_R2_R3_to_R4', 'loro_R2_R4_to_R3', 'loro_R3_R4_to_R2',
            'joint_R2_R3_R4',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6', 'jitter_R2_R3_R4_to_R7'
        ]

    analyzer = ErrorCorrelationAnalyzer(args.results_root)
    analyzer.generate_report(
        tasks=args.tasks,
        models=args.models,
        feature_set=args.feature_set,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
