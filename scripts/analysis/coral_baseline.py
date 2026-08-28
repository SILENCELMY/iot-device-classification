#!/usr/bin/env python3
"""
CORAL (CORrelation ALignment) Baseline Implementation
目标: 缓解 feature covariance drift, 验证是否能降低 confusion pattern drift
方法: 对齐源域和目标域的特征协方差矩阵
参考: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
import json


class CORALAligner:
    """CORAL 特征对齐器"""

    def __init__(self):
        self.mean_src = None
        self.cov_src = None
        self.mean_tgt = None
        self.cov_tgt = None
        self.whitening_mat = None  # 源域白化矩阵
        self.coloring_mat = None   # 目标域着色矩阵

    def fit(self, X_src: np.ndarray, X_tgt: np.ndarray):
        """
        学习 CORAL 变换
        X_src: 源域特征 (n_src, n_features)
        X_tgt: 目标域特征 (n_tgt, n_features) - 无标签
        """
        # 中心化
        self.mean_src = X_src.mean(axis=0, keepdims=True)
        self.mean_tgt = X_tgt.mean(axis=0, keepdims=True)

        X_src_centered = X_src - self.mean_src
        X_tgt_centered = X_tgt - self.mean_tgt

        # 计算协方差矩阵
        self.cov_src = np.cov(X_src_centered.T) + np.eye(X_src.shape[1]) * 1e-5  # 正则化
        self.cov_tgt = np.cov(X_tgt_centered.T) + np.eye(X_tgt.shape[1]) * 1e-5

        # CORAL 变换: X_aligned = (X_src - mean_src) @ Cs^{-1/2} @ Ct^{1/2} + mean_tgt
        # Cs^{-1/2}: 源域白化 (decorrelate)
        # Ct^{1/2}: 目标域着色 (re-correlate to target)

        # 白化矩阵 (Cholesky 分解)
        try:
            L_src = np.linalg.cholesky(self.cov_src)  # Cs = L_src @ L_src.T
            self.whitening_mat = np.linalg.inv(L_src.T)  # Cs^{-1/2}
        except np.linalg.LinAlgError:
            # 如果 Cholesky 失败，使用 SVD
            U_src, S_src, _ = np.linalg.svd(self.cov_src)
            self.whitening_mat = U_src @ np.diag(1.0 / np.sqrt(S_src + 1e-5)) @ U_src.T

        # 着色矩阵
        try:
            L_tgt = np.linalg.cholesky(self.cov_tgt)  # Ct = L_tgt @ L_tgt.T
            self.coloring_mat = L_tgt  # Ct^{1/2}
        except np.linalg.LinAlgError:
            U_tgt, S_tgt, _ = np.linalg.svd(self.cov_tgt)
            self.coloring_mat = U_tgt @ np.diag(np.sqrt(S_tgt + 1e-5)) @ U_tgt.T

        return self

    def transform(self, X: np.ndarray, mode: str = 'source') -> np.ndarray:
        """
        应用 CORAL 变换
        mode: 'source' (对齐源域到目标域) 或 'target' (对齐目标域到源域)
        """
        if mode == 'source':
            # 源域对齐到目标域
            X_centered = X - self.mean_src
            X_aligned = X_centered @ self.whitening_mat @ self.coloring_mat + self.mean_tgt
        elif mode == 'target':
            # 目标域对齐到源域 (如果需要的话)
            X_centered = X - self.mean_tgt
            # 反向变换
            U_tgt, S_tgt, _ = np.linalg.svd(self.cov_tgt)
            inv_coloring = U_tgt @ np.diag(1.0 / np.sqrt(S_tgt + 1e-5)) @ U_tgt.T
            U_src, S_src, _ = np.linalg.svd(self.cov_src)
            coloring_src = U_src @ np.diag(np.sqrt(S_src + 1e-5)) @ U_src.T
            X_aligned = X_centered @ inv_coloring @ coloring_src + self.mean_src
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return X_aligned


class CORALRFBaseline:
    """CORAL + RF Baseline 实验"""

    def __init__(self, results_root: str, feature_cache_path: str):
        self.results_root = Path(results_root)
        self.feature_cache_path = Path(feature_cache_path)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']

    def load_features_and_labels(self) -> pd.DataFrame:
        """加载特征缓存"""
        if not self.feature_cache_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {self.feature_cache_path}")
        return pd.read_csv(self.feature_cache_path)

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        train_rounds: list,
        test_rounds: list
    ) -> tuple:
        """准备训练/测试集"""
        train_mask = df['round'].isin(train_rounds)
        test_mask = df['round'].isin(test_rounds)

        # 特征列 (排除 metadata)
        meta_cols = ['round', 'traffic', 'filter_mode', 'source_file', 'window_id',
                     'window_start', 'window_end', 'label']
        feature_cols = [c for c in df.columns if c not in meta_cols]

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'label'].values

        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, 'label'].values

        return X_train, y_train, X_test, y_test, feature_cols

    def run_loro_experiment(
        self,
        train_rounds: list,
        test_round: str,
        use_coral: bool = False,
        random_state: int = 42
    ) -> dict:
        """
        运行 LORO 实验
        use_coral: 是否使用 CORAL 对齐
        """
        df = self.load_features_and_labels()

        X_train, y_train, X_test, y_test, feature_cols = self.prepare_train_test_split(
            df, train_rounds, [test_round]
        )

        print(f"\n{'='*80}")
        print(f"LORO: Train on {train_rounds} → Test on {test_round}")
        print(f"CORAL: {'Enabled' if use_coral else 'Disabled'}")
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"{'='*80}")

        # CORAL 对齐 (如果启用)
        if use_coral:
            print("Applying CORAL alignment...")
            coral = CORALAligner()
            coral.fit(X_train, X_test)
            X_train_aligned = coral.transform(X_train, mode='source')

            # 检查对齐效果
            cov_train_before = np.cov(X_train.T)
            cov_train_after = np.cov(X_train_aligned.T)
            cov_test = np.cov(X_test.T)

            cov_diff_before = np.linalg.norm(cov_train_before - cov_test, ord='fro')
            cov_diff_after = np.linalg.norm(cov_train_after - cov_test, ord='fro')

            print(f"  Covariance diff (before CORAL): {cov_diff_before:.3f}")
            print(f"  Covariance diff (after CORAL):  {cov_diff_after:.3f}")
            print(f"  Reduction: {(1 - cov_diff_after/cov_diff_before)*100:.1f}%")

            X_train_final = X_train_aligned
        else:
            X_train_final = X_train

        # 训练 RF
        print("Training Random Forest...")
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        clf.fit(X_train_final, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 计算指标
        cm = confusion_matrix(y_test, y_pred, labels=self.class_names)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        report = classification_report(y_test, y_pred, labels=self.class_names,
                                     output_dict=True, zero_division=0)

        macro_f1 = f1_score(y_test, y_pred, labels=self.class_names, average='macro', zero_division=0)
        accuracy = (y_pred == y_test).mean()

        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro-F1: {macro_f1:.4f}")

        # 逐类 F1
        print(f"\n  Per-class F1:")
        for class_name in self.class_names:
            f1 = report[class_name]['f1-score']
            print(f"    {class_name:<12}: {f1:.4f}")

        return {
            'train_rounds': train_rounds,
            'test_round': test_round,
            'use_coral': use_coral,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'classification_report': report,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'cov_reduction': (1 - cov_diff_after/cov_diff_before)*100 if use_coral else 0.0
        }

    def compare_coral_vs_baseline(
        self,
        loro_configs: list,
        output_dir: str = None
    ) -> dict:
        """
        对比 CORAL vs Baseline 在多个 LORO 任务上的表现
        loro_configs: [(train_rounds, test_round), ...]
        """
        if output_dir is None:
            output_dir = self.results_root.parent / 'report'
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results_baseline = []
        results_coral = []

        for train_rounds, test_round in loro_configs:
            # Baseline (no CORAL)
            res_baseline = self.run_loro_experiment(train_rounds, test_round, use_coral=False)
            results_baseline.append(res_baseline)

            # CORAL
            res_coral = self.run_loro_experiment(train_rounds, test_round, use_coral=True)
            results_coral.append(res_coral)

        # 可视化对比
        self.visualize_comparison(results_baseline, results_coral, loro_configs, output_dir)

        # 生成报告
        self.generate_comparison_report(results_baseline, results_coral, loro_configs, output_dir)

        return {
            'baseline': results_baseline,
            'coral': results_coral
        }

    def visualize_comparison(
        self,
        results_baseline: list,
        results_coral: list,
        configs: list,
        output_dir: Path
    ):
        """可视化 CORAL vs Baseline 对比"""
        n_tasks = len(configs)

        # 1. Macro-F1 对比
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1a. Macro-F1 柱状图
        ax1 = axes[0]
        task_names = [f"{'-'.join(train)}→{test}" for train, test in configs]
        x = np.arange(n_tasks)
        width = 0.35

        f1_baseline = [r['macro_f1'] for r in results_baseline]
        f1_coral = [r['macro_f1'] for r in results_coral]

        ax1.bar(x - width/2, f1_baseline, width, label='Baseline (RF)', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, f1_coral, width, label='CORAL + RF', color='coral', alpha=0.8)

        ax1.set_xlabel('LORO Task', fontsize=12)
        ax1.set_ylabel('Macro-F1', fontsize=12)
        ax1.set_title('CORAL vs Baseline: Macro-F1 Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(task_names, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 标注改进幅度
        for i in range(n_tasks):
            delta = f1_coral[i] - f1_baseline[i]
            y_pos = max(f1_baseline[i], f1_coral[i]) + 0.02
            color = 'green' if delta > 0 else 'red'
            ax1.text(i, y_pos, f'{delta:+.3f}', ha='center', fontsize=9, color=color, fontweight='bold')

        # 1b. Covariance reduction vs F1 improvement
        ax2 = axes[1]
        cov_reductions = [r['cov_reduction'] for r in results_coral]
        f1_improvements = [(f1_coral[i] - f1_baseline[i]) * 100 for i in range(n_tasks)]

        ax2.scatter(cov_reductions, f1_improvements, s=100, alpha=0.7, edgecolors='black')
        for i, task_name in enumerate(task_names):
            ax2.annotate(task_name, (cov_reductions[i], f1_improvements[i]),
                        fontsize=8, ha='center', xytext=(0, 5), textcoords='offset points')

        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Covariance Reduction (%)', fontsize=12)
        ax2.set_ylabel('F1 Improvement (%)', fontsize=12)
        ax2.set_title('Covariance Alignment vs Performance Gain', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'coral_vs_baseline_f1.png', dpi=300, bbox_inches='tight')
        print(f"✅ F1 comparison plot saved: coral_vs_baseline_f1.png")

        # 2. 混淆矩阵对比 (选择 worst case)
        worst_idx = np.argmin(f1_baseline)
        worst_task = configs[worst_idx]

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

        cm_baseline = results_baseline[worst_idx]['confusion_matrix_normalized']
        cm_coral = results_coral[worst_idx]['confusion_matrix_normalized']

        sns.heatmap(cm_baseline, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes2[0], vmin=0, vmax=1)
        axes2[0].set_title(f'Baseline (F1={f1_baseline[worst_idx]:.3f})', fontsize=12, fontweight='bold')
        axes2[0].set_xlabel('Predicted', fontsize=10)
        axes2[0].set_ylabel('True', fontsize=10)

        sns.heatmap(cm_coral, annot=True, fmt='.2f', cmap='Oranges',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes2[1], vmin=0, vmax=1)
        axes2[1].set_title(f'CORAL (F1={f1_coral[worst_idx]:.3f})', fontsize=12, fontweight='bold')
        axes2[1].set_xlabel('Predicted', fontsize=10)
        axes2[1].set_ylabel('True', fontsize=10)

        fig2.suptitle(f'Confusion Matrix: {task_names[worst_idx]} (Worst Case)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'coral_vs_baseline_confusion_worst.png', dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix comparison saved: coral_vs_baseline_confusion_worst.png")

    def generate_comparison_report(
        self,
        results_baseline: list,
        results_coral: list,
        configs: list,
        output_dir: Path
    ):
        """生成 CORAL vs Baseline 对比报告"""
        report_lines = []
        report_lines.append("# CORAL + RF Baseline Comparison\n")
        report_lines.append("**Method**: CORrelation ALignment (CORAL) for feature covariance alignment  \n")
        report_lines.append("**Baseline**: Random Forest without domain adaptation  \n\n")

        report_lines.append("## 1. Overall Performance Comparison\n")
        report_lines.append("![F1 Comparison](coral_vs_baseline_f1.png)\n")

        report_lines.append("## 2. Worst Case Confusion Matrix\n")
        report_lines.append("![Confusion Matrix](coral_vs_baseline_confusion_worst.png)\n")

        report_lines.append("## 3. Detailed Results\n")
        report_lines.append("| LORO Task | Baseline F1 | CORAL F1 | Δ F1 | Cov Reduction (%) |\n")
        report_lines.append("|---|---|---|---|---|\n")

        for i, (train_rounds, test_round) in enumerate(configs):
            task_name = f"{'-'.join(train_rounds)}→{test_round}"
            f1_base = results_baseline[i]['macro_f1']
            f1_coral = results_coral[i]['macro_f1']
            delta = f1_coral - f1_base
            cov_red = results_coral[i]['cov_reduction']

            delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
            report_lines.append(f"| {task_name} | {f1_base:.4f} | {f1_coral:.4f} | {delta_str} | {cov_red:.1f} |\n")

        # 统计摘要
        f1_improvements = [(results_coral[i]['macro_f1'] - results_baseline[i]['macro_f1']) * 100
                          for i in range(len(configs))]
        mean_improvement = np.mean(f1_improvements)
        std_improvement = np.std(f1_improvements)
        positive_count = sum(1 for imp in f1_improvements if imp > 0)

        report_lines.append("\n## 4. Summary Statistics\n")
        report_lines.append(f"- **Mean F1 improvement**: {mean_improvement:+.2f}% (± {std_improvement:.2f}%)\n")
        report_lines.append(f"- **Tasks improved**: {positive_count}/{len(configs)}\n")

        # Format task names separately to avoid f-string issues
        task_names = [f"{'-'.join(train)}→{test}" for train, test in configs]
        best_task = task_names[np.argmax(f1_improvements)]
        worst_task = task_names[np.argmin(f1_improvements)]

        report_lines.append(f"- **Best improvement**: {max(f1_improvements):+.2f}% on {best_task}\n")
        report_lines.append(f"- **Worst change**: {min(f1_improvements):+.2f}% on {worst_task}\n")

        report_lines.append("\n## 5. Key Findings\n")

        if mean_improvement > 0:
            report_lines.append(f"- ✅ **CORAL improves cross-environment generalization** on average ({mean_improvement:+.2f}%)\n")
        else:
            report_lines.append(f"- ❌ **CORAL does NOT improve performance** on average ({mean_improvement:+.2f}%)\n")

        report_lines.append(f"- Covariance alignment **reduces distribution shift**, but...\n")

        if mean_improvement < 1.0:
            report_lines.append(f"- **Limited performance gain** → covariance drift is **not the main bottleneck**\n")
            report_lines.append(f"- **Implication**: The problem is more about **confusion pattern drift** (class-conditional shift) rather than marginal feature covariance shift\n")

        report_lines.append("\n## 6. Implications for Robust IoT Device Classification\n")
        report_lines.append("- **Feature covariance alignment alone is insufficient**\n")
        report_lines.append("- **Need class-conditional alignment** or **confusion-aware adaptation**\n")
        report_lines.append("- **Future directions**: \n")
        report_lines.append("  1. Conditional CORAL (align per-class covariances)\n")
        report_lines.append("  2. Adversarial domain adaptation (align class-conditional distributions)\n")
        report_lines.append("  3. Confusion-aware meta-learning\n")

        # 保存报告
        report_path = output_dir / 'coral_baseline_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        print(f"✅ Report saved: {report_path}")

        # 保存数值结果
        results_df = pd.DataFrame([
            {
                'task': f"{'-'.join(train)}→{test}",
                'baseline_f1': results_baseline[i]['macro_f1'],
                'coral_f1': results_coral[i]['macro_f1'],
                'delta_f1': results_coral[i]['macro_f1'] - results_baseline[i]['macro_f1'],
                'cov_reduction_pct': results_coral[i]['cov_reduction']
            }
            for i, (train, test) in enumerate(configs)
        ])
        results_df.to_csv(output_dir / 'coral_baseline_results.csv', index=False)
        print(f"✅ Results CSV saved: coral_baseline_results.csv")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CORAL + RF Baseline Experiment')
    parser.add_argument('--results-root', type=str, required=True)
    parser.add_argument('--feature-cache', type=str, required=True,
                        help='Path to feature cache CSV')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    # LORO 配置
    loro_configs = [
        (['R2', 'R3'], 'R4'),
        (['R2', 'R4'], 'R3'),
        (['R3', 'R4'], 'R2'),
    ]

    baseline = CORALRFBaseline(args.results_root, args.feature_cache)
    baseline.compare_coral_vs_baseline(loro_configs, args.output_dir)

    print("\n" + "=" * 80)
    print("CORAL Baseline Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
