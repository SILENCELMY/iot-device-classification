#!/usr/bin/env python3
"""
Meta-Feature Distribution Shift Analysis
目标: 分析 stacking 输入 (base model probability vectors) 的分布漂移
方法: 计算 meta-feature space 的统计特性，并比较跨环境的差异
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import sys

CORE_DIR = Path(__file__).resolve().parents[1] / 'core'
sys.path.insert(0, str(CORE_DIR))


class MetaFeatureAnalyzer:
    """分析 Stacking meta-features 的分布漂移"""

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']

    def load_stacking_model(self, task: str, feature_set: str = 'all_features'):
        """加载 stacking 模型以提取 base model predictions"""
        import __main__
        from robust_iot_research import SimpleStackingClassifier
        __main__.SimpleStackingClassifier = SimpleStackingClassifier

        import joblib
        model_path = self.results_root / task / feature_set / 'stacking' / 'model.joblib'
        if not model_path.exists():
            return None
        return joblib.load(model_path)

    def extract_meta_features_from_predictions(
        self,
        task: str,
        base_models: list = None,
        feature_set: str = 'all_features'
    ) -> tuple:
        """
        从预测文件提取 meta-features (base model probability vectors)
        但由于我们只有最终预测，需要重新推理

        更简单的方法: 直接从各 base model 的预测置信度构造 meta-features
        但 predictions.csv 只有 predicted_label, 没有概率

        因此采用替代方案: 从 confusion matrix 估计类别概率分布
        """
        # 读取各 base model 的预测
        predictions = {}
        for model in base_models:
            pred_path = self.results_root / task / feature_set / model / 'predictions.csv'
            if pred_path.exists():
                predictions[model] = pd.read_csv(pred_path)

        if len(predictions) == 0:
            return None, None

        # 构造 meta-features: one-hot encoding of predicted labels
        # Shape: (n_samples, n_base_models * n_classes)
        n_samples = len(predictions[base_models[0]])
        n_classes = len(self.class_names)
        meta_features = np.zeros((n_samples, len(base_models) * n_classes))

        for i, model in enumerate(base_models):
            pred_df = predictions[model]
            for j, class_name in enumerate(self.class_names):
                # One-hot: 预测为该类则为 1
                meta_features[:, i * n_classes + j] = (pred_df['predicted_label'] == class_name).astype(float)

        # 真实标签
        true_labels = predictions[base_models[0]]['true_label'].values

        return meta_features, true_labels

    def compute_distribution_shift_metrics(
        self,
        meta_features_1: np.ndarray,
        meta_features_2: np.ndarray,
        metric: str = 'wasserstein'
    ) -> dict:
        """
        计算两个 meta-feature 分布之间的距离
        metric: 'wasserstein', 'kl', 'mmd' (Maximum Mean Discrepancy)
        """
        results = {}

        # 逐维计算 Wasserstein distance
        n_dims = meta_features_1.shape[1]
        wasserstein_dists = []

        for dim in range(n_dims):
            dist = wasserstein_distance(meta_features_1[:, dim], meta_features_2[:, dim])
            wasserstein_dists.append(dist)

        results['wasserstein_per_dim'] = wasserstein_dists
        results['wasserstein_mean'] = np.mean(wasserstein_dists)
        results['wasserstein_max'] = np.max(wasserstein_dists)

        # KS test (Kolmogorov-Smirnov)
        ks_stats = []
        ks_pvals = []
        for dim in range(n_dims):
            stat, pval = ks_2samp(meta_features_1[:, dim], meta_features_2[:, dim])
            ks_stats.append(stat)
            ks_pvals.append(pval)

        results['ks_stat_per_dim'] = ks_stats
        results['ks_stat_mean'] = np.mean(ks_stats)
        results['ks_pval_min'] = np.min(ks_pvals)  # 最显著的差异

        # MMD (Maximum Mean Discrepancy) - simplified version
        mean_1 = meta_features_1.mean(axis=0)
        mean_2 = meta_features_2.mean(axis=0)
        mmd = np.linalg.norm(mean_1 - mean_2)
        results['mmd'] = mmd

        # Covariance difference
        cov_1 = np.cov(meta_features_1.T)
        cov_2 = np.cov(meta_features_2.T)
        cov_diff = np.linalg.norm(cov_1 - cov_2, ord='fro')
        results['cov_frobenius'] = cov_diff

        return results

    def visualize_meta_feature_space(
        self,
        tasks: list,
        base_models: list,
        feature_set: str = 'all_features',
        output_path: str = None
    ):
        """使用 PCA/t-SNE 可视化 meta-feature space 的分布差异"""
        meta_features_all = []
        labels_all = []
        task_labels = []

        # 收集所有任务的 meta-features
        for task in tasks:
            mf, true_labels = self.extract_meta_features_from_predictions(task, base_models, feature_set)
            if mf is not None:
                meta_features_all.append(mf)
                labels_all.append(true_labels)
                task_labels.extend([task] * len(mf))

        if len(meta_features_all) == 0:
            print("No meta-features found")
            return

        # 合并
        meta_features_all = np.vstack(meta_features_all)
        labels_all = np.concatenate(labels_all)

        # PCA 降维
        pca = PCA(n_components=2)
        mf_pca = pca.fit_transform(meta_features_all)

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 按任务着色
        ax1 = axes[0]
        task_colors = {}
        from matplotlib import colormaps
        cmap = colormaps['tab10']
        for i, task in enumerate(tasks):
            task_colors[task] = cmap(i % 10)

        for task in tasks:
            mask = np.array(task_labels) == task
            ax1.scatter(mf_pca[mask, 0], mf_pca[mask, 1],
                       label=task, alpha=0.5, s=20, color=task_colors[task])

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
        ax1.set_title('Meta-Feature Space by Task', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=8, loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)

        # 2. 按真实类别着色
        ax2 = axes[1]
        class_colors = {'Camera': '#1f77b4', 'Light_T1': '#ff7f0e', 'Light_XM': '#2ca02c',
                       'Sensor': '#d62728', 'Socket': '#9467bd'}

        for class_name in self.class_names:
            mask = labels_all == class_name
            ax2.scatter(mf_pca[mask, 0], mf_pca[mask, 1],
                       label=class_name, alpha=0.5, s=20, color=class_colors.get(class_name, 'gray'))

        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
        ax2.set_title('Meta-Feature Space by Class', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Meta-feature space visualization saved: {output_path}")

    def analyze_per_class_meta_distribution(
        self,
        task1: str,
        task2: str,
        base_models: list,
        feature_set: str = 'all_features'
    ) -> dict:
        """逐类分析 meta-feature 分布差异"""
        mf1, labels1 = self.extract_meta_features_from_predictions(task1, base_models, feature_set)
        mf2, labels2 = self.extract_meta_features_from_predictions(task2, base_models, feature_set)

        if mf1 is None or mf2 is None:
            return None

        results = {}

        for class_name in self.class_names:
            mask1 = labels1 == class_name
            mask2 = labels2 == class_name

            if mask1.sum() == 0 or mask2.sum() == 0:
                continue

            class_mf1 = mf1[mask1]
            class_mf2 = mf2[mask2]

            # 计算分布差异
            metrics = self.compute_distribution_shift_metrics(class_mf1, class_mf2)
            results[class_name] = metrics

        return results

    def compute_shift_matrix(
        self,
        tasks: list,
        base_models: list,
        feature_set: str = 'all_features'
    ) -> tuple:
        """计算任务间的 meta-feature 分布漂移矩阵"""
        n = len(tasks)
        wasserstein_matrix = np.zeros((n, n))
        mmd_matrix = np.zeros((n, n))

        # 加载所有 meta-features
        meta_features_dict = {}
        for task in tasks:
            mf, _ = self.extract_meta_features_from_predictions(task, base_models, feature_set)
            if mf is not None:
                meta_features_dict[task] = mf

        # 计算成对距离
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if task1 not in meta_features_dict or task2 not in meta_features_dict:
                    wasserstein_matrix[i, j] = np.nan
                    mmd_matrix[i, j] = np.nan
                    continue

                mf1 = meta_features_dict[task1]
                mf2 = meta_features_dict[task2]

                metrics = self.compute_distribution_shift_metrics(mf1, mf2)
                wasserstein_matrix[i, j] = metrics['wasserstein_mean']
                mmd_matrix[i, j] = metrics['mmd']

        return wasserstein_matrix, mmd_matrix, meta_features_dict

    def generate_report(
        self,
        tasks: list,
        base_models: list = None,
        feature_set: str = 'all_features',
        output_dir: str = None
    ):
        """生成完整的 meta-feature 分布漂移分析报告"""
        if base_models is None:
            base_models = ['rf', 'xgboost', 'lightgbm']

        if output_dir is None:
            output_dir = self.results_root.parent / 'report'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print("Meta-Feature Distribution Shift Analysis")
        print("=" * 80)

        # 1. 计算漂移矩阵
        wasserstein_matrix, mmd_matrix, meta_features_dict = self.compute_shift_matrix(
            tasks, base_models, feature_set
        )

        # 2. 可视化漂移矩阵
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax1 = axes[0]
        sns.heatmap(wasserstein_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=tasks, yticklabels=tasks, ax=ax1,
                   cbar_kws={'label': 'Wasserstein Distance'})
        ax1.set_title('Meta-Feature Distribution Shift (Wasserstein)', fontsize=14, fontweight='bold')

        ax2 = axes[1]
        sns.heatmap(mmd_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=tasks, yticklabels=tasks, ax=ax2,
                   cbar_kws={'label': 'MMD'})
        ax2.set_title('Meta-Feature Distribution Shift (MMD)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'meta_feature_shift_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Meta-feature shift matrix saved: meta_feature_shift_matrix.png")

        # 3. 可视化 meta-feature space
        self.visualize_meta_feature_space(tasks, base_models, feature_set,
                                         output_dir / 'meta_feature_space_pca.png')

        # 4. 逐类分析 (LORO 崩盘 vs IID)
        print("\n分析 LORO 崩盘场景 vs IID 场景的 meta-feature 分布差异...")
        loro_collapse = 'loro_R2_R4_to_R3'
        iid_task = 'single_round_R3'

        class_results = self.analyze_per_class_meta_distribution(
            loro_collapse, iid_task, base_models, feature_set
        )

        if class_results:
            class_df = pd.DataFrame([
                {
                    'class': class_name,
                    'wasserstein': metrics['wasserstein_mean'],
                    'mmd': metrics['mmd'],
                    'ks_stat': metrics['ks_stat_mean']
                }
                for class_name, metrics in class_results.items()
            ])
            class_df.to_csv(output_dir / 'meta_feature_shift_per_class.csv', index=False)
            print(f"✅ Per-class meta-feature shift saved: meta_feature_shift_per_class.csv")

        # 5. 生成 markdown 报告
        report_lines = []
        report_lines.append("# Meta-Feature Distribution Shift Analysis\n")
        report_lines.append(f"**Base Models**: {', '.join([m.upper() for m in base_models])}  ")
        report_lines.append(f"**Feature Set**: {feature_set}  \n")

        report_lines.append("## 1. Distribution Shift Matrix\n")
        report_lines.append("![Shift Matrix](meta_feature_shift_matrix.png)\n")

        report_lines.append("## 2. Meta-Feature Space Visualization (PCA)\n")
        report_lines.append("![PCA Space](meta_feature_space_pca.png)\n")

        report_lines.append("## 3. Key Findings\n")

        # 找到最大和最小漂移
        n = len(tasks)
        max_shift = -1
        min_shift = np.inf
        max_pair = None
        min_pair = None

        for i in range(n):
            for j in range(i+1, n):
                if not np.isnan(wasserstein_matrix[i, j]):
                    shift = wasserstein_matrix[i, j]
                    if shift > max_shift:
                        max_shift = shift
                        max_pair = (tasks[i], tasks[j])
                    if shift < min_shift:
                        min_shift = shift
                        min_pair = (tasks[i], tasks[j])

        report_lines.append(f"### Largest Meta-Feature Distribution Shift:\n")
        if max_pair:
            report_lines.append(f"- **{max_pair[0]} ↔ {max_pair[1]}**: Wasserstein = {max_shift:.3f}\n")
            report_lines.append(f"  - **Implication**: Stacking's meta-learner sees **very different input distributions**\n")

        report_lines.append(f"\n### Smallest Meta-Feature Distribution Shift:\n")
        if min_pair:
            report_lines.append(f"- **{min_pair[0]} ↔ {min_pair[1]}**: Wasserstein = {min_shift:.3f}\n")
            report_lines.append(f"  - **Implication**: Meta-learner input distributions are **similar** → easier generalization\n")

        # LORO vs IID 对比
        if class_results:
            report_lines.append(f"\n## 4. Per-Class Analysis: LORO Collapse vs IID\n")
            report_lines.append(f"**Comparison**: `{loro_collapse}` (collapse) vs `{iid_task}` (IID)  \n")
            report_lines.append("| Class | Wasserstein | MMD | KS Stat |\n")
            report_lines.append("|---|---|---|---|\n")

            for _, row in class_df.iterrows():
                report_lines.append(f"| {row['class']} | {row['wasserstein']:.3f} | {row['mmd']:.3f} | {row['ks_stat']:.3f} |\n")

            # 哪个类别漂移最严重
            max_shift_class = class_df.loc[class_df['wasserstein'].idxmax()]
            report_lines.append(f"\n**Most Shifted Class**: {max_shift_class['class']} (Wasserstein = {max_shift_class['wasserstein']:.3f})  \n")
            report_lines.append(f"**Interpretation**: Meta-learner sees **most different base-model predictions** for this class  \n")

        report_lines.append("\n## 5. Implications for Stacking Failure\n")
        report_lines.append("- **High meta-feature shift** → meta-learner's training distribution ≠ test distribution\n")
        report_lines.append("- **Meta-learner overfits** to training-time base-model prediction patterns\n")
        report_lines.append("- **Test-time base predictions** have different statistical properties → meta-learner fails\n")

        # 保存报告
        report_path = output_dir / 'meta_feature_shift_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"✅ Report saved: {report_path}")

        return {
            'wasserstein_matrix': wasserstein_matrix,
            'mmd_matrix': mmd_matrix,
            'meta_features_dict': meta_features_dict
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Meta-Feature Distribution Shift Analysis')
    parser.add_argument('--results-root', type=str, required=True)
    parser.add_argument('--base-models', type=str, nargs='+', default=['rf', 'xgboost', 'lightgbm'])
    parser.add_argument('--feature-set', type=str, default='all_features')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--tasks', type=str, nargs='+', default=None)

    args = parser.parse_args()

    if args.tasks is None:
        args.tasks = [
            'single_round_R2', 'single_round_R3', 'single_round_R4',
            'loro_R2_R3_to_R4', 'loro_R2_R4_to_R3', 'loro_R3_R4_to_R2',
            'joint_R2_R3_R4',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6', 'jitter_R2_R3_R4_to_R7'
        ]

    analyzer = MetaFeatureAnalyzer(args.results_root)
    analyzer.generate_report(
        tasks=args.tasks,
        base_models=args.base_models,
        feature_set=args.feature_set,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
