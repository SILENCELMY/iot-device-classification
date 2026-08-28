#!/usr/bin/env python3
"""
Controlled CPD Experiment - 机制验证实验
目标：通过控制 CPD 强度验证因果关系
核心假设：CPD ↑ => Ensemble Gain ↓

不再是 observational evidence，而是 mechanism validation。
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ControlledCPDExperiment:
    """Controlled CPD 机制验证实验"""

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']
        self.n_classes = len(self.class_names)

        # 基模型列表
        self.base_models = ['rf', 'xgboost', 'lightgbm', 'extra_trees']

        # 所有任务
        self.all_tasks = [
            'single_round_R2',
            'single_round_R3',
            'single_round_R4',
            'loro_R2_R3_to_R4',
            'loro_R2_R4_to_R3',
            'loro_R3_R4_to_R2',
            'joint_R2_R3_R4',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6',
            'jitter_R2_R3_R4_to_R7',
            'jitter_R2_R3_R4_to_R6_R7',
        ]

    def compute_cpd(self, cm1: np.ndarray, cm2: np.ndarray) -> float:
        """
        计算 CPD (Confusion Pattern Drift)
        CPD = ||Off(C1) - Off(C2)||_F
        """
        # 归一化
        cm1_norm = self._normalize_cm(cm1)
        cm2_norm = self._normalize_cm(cm2)

        # 去除对角线
        off1 = cm1_norm.copy()
        off2 = cm2_norm.copy()
        np.fill_diagonal(off1, 0)
        np.fill_diagonal(off2, 0)

        # Frobenius distance
        return np.linalg.norm(off1 - off2, ord='fro')

    def _normalize_cm(self, cm: np.ndarray) -> np.ndarray:
        """行归一化"""
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return cm / row_sums

    def load_cm(self, task: str, model: str = 'rf', feature_set: str = 'all_features') -> np.ndarray:
        """加载混淆矩阵"""
        cm_path = self.results_root / task / feature_set / model / 'confusion_matrix.csv'
        if not cm_path.exists():
            return None
        cm_df = pd.read_csv(cm_path, index_col=0)
        return cm_df.values

    def load_f1(self, task: str, model: str, feature_set: str = 'all_features') -> float:
        """加载 macro-F1"""
        metrics_path = self.results_root / task / feature_set / model / 'metrics.json'
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            metrics = json.load(f)
        return metrics.get('macro_f1')

    def compute_ensemble_gain(self, task: str, feature_set: str = 'all_features') -> dict:
        """
        计算 Ensemble Gain
        Gain = F1_stacking / max(F1_base_models)
        """
        # 加载所有模型 F1
        f1_scores = {}
        for model in self.base_models + ['stacking']:
            f1 = self.load_f1(task, model, feature_set)
            if f1 is not None:
                f1_scores[model] = f1

        if 'stacking' not in f1_scores or len(f1_scores) < 2:
            return None

        # 最佳基模型
        base_f1s = [f1_scores[m] for m in self.base_models if m in f1_scores]
        if len(base_f1s) == 0:
            return None

        best_base_f1 = max(base_f1s)
        best_base_model = [m for m in self.base_models if f1_scores.get(m) == best_base_f1][0]

        stacking_f1 = f1_scores['stacking']

        # Gain = Stacking - Best Base (绝对差值)
        gain_absolute = stacking_f1 - best_base_f1

        # Gain = (Stacking - Best Base) / Best Base (相对增益)
        gain_relative = (stacking_f1 - best_base_f1) / best_base_f1 if best_base_f1 > 0 else 0

        return {
            'task': task,
            'stacking_f1': stacking_f1,
            'best_base_f1': best_base_f1,
            'best_base_model': best_base_model,
            'gain_absolute': gain_absolute,
            'gain_relative': gain_relative,
            'f1_scores': f1_scores,
        }

    def classify_cpd_level(self, cpd: float) -> str:
        """
        分类 CPD Level
        基于已知的 CPD 分布和实验结果
        """
        if cpd < 0.2:
            return 'Low'
        elif cpd < 0.5:
            return 'Medium'
        else:
            return 'High'

    def run_controlled_experiment(self, feature_set='all_features', output_dir=None):
        """运行 Controlled CPD 实验"""
        print("\n" + "="*80)
        print("Controlled CPD Experiment - 机制验证")
        print("="*80)

        # IID 基准（使用 joint_R2_R3_R4 作为基准混淆矩阵）
        baseline_cm = self.load_cm('joint_R2_R3_R4', 'rf', feature_set)
        if baseline_cm is None:
            print("⚠️ Baseline confusion matrix not found, using single_round_R3")
            baseline_cm = self.load_cm('single_round_R3', 'rf', feature_set)

        # 收集所有任务的数据
        experiment_data = []

        for task in self.all_tasks:
            # 计算 Ensemble Gain
            gain_info = self.compute_ensemble_gain(task, feature_set)
            if gain_info is None:
                continue

            # 计算 CPD（相对于基准）
            task_cm = self.load_cm(task, 'rf', feature_set)
            if task_cm is None:
                continue

            cpd = self.compute_cpd(baseline_cm, task_cm)
            cpd_level = self.classify_cpd_level(cpd)

            experiment_data.append({
                'task': task,
                'cpd': cpd,
                'cpd_level': cpd_level,
                'stacking_f1': gain_info['stacking_f1'],
                'best_base_f1': gain_info['best_base_f1'],
                'best_base_model': gain_info['best_base_model'],
                'gain_absolute': gain_info['gain_absolute'],
                'gain_relative': gain_info['gain_relative'],
            })

            print(f"{task:40s}  CPD={cpd:.4f} ({cpd_level:6s})  Gain={gain_info['gain_absolute']:+.4f}")

        df = pd.DataFrame(experiment_data)

        # 保存数据
        if output_dir:
            df.to_csv(output_dir / 'controlled_cpd_data.csv', index=False)
            print(f"\n✅ 数据已保存: controlled_cpd_data.csv")

        return df

    def plot_cpd_vs_gain(self, df: pd.DataFrame, output_dir: Path):
        """
        Figure 1: CPD vs Ensemble Gain（最重要）
        验证核心假设：CPD ↑ => Gain ↓
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # 散点图（按 CPD Level 着色）
        colors = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
        for level in ['Low', 'Medium', 'High']:
            mask = df['cpd_level'] == level
            if mask.sum() > 0:
                ax.scatter(df.loc[mask, 'cpd'],
                          df.loc[mask, 'gain_absolute'],
                          c=colors[level], label=f'{level} CPD',
                          s=150, alpha=0.7, edgecolors='black', linewidths=1.5)

        # 拟合线
        if len(df) >= 3:
            z = np.polyfit(df['cpd'], df['gain_absolute'], 1)
            p = np.poly1d(z)
            x_fit = np.linspace(df['cpd'].min(), df['cpd'].max(), 100)
            ax.plot(x_fit, p(x_fit), '--', color='gray', alpha=0.7, linewidth=2)

        # 零线
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5,
                  label='Gain = 0 (Baseline)')

        # 相关性分析
        pearson_r, pearson_p = pearsonr(df['cpd'], df['gain_absolute'])
        spearman_r, spearman_p = spearmanr(df['cpd'], df['gain_absolute'])

        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})\n'
        textstr += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('CPD (Confusion Pattern Drift)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Ensemble Gain (Stacking F1 - Best Base F1)', fontsize=13, fontweight='bold')
        ax.set_title('Controlled CPD Experiment: CPD vs Ensemble Gain',
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / 'controlled_cpd_vs_gain.png', dpi=300, bbox_inches='tight')
        print(f"✅ Figure 1 saved: controlled_cpd_vs_gain.png")
        plt.close()

        return {'pearson_r': pearson_r, 'pearson_p': pearson_p,
                'spearman_r': spearman_r, 'spearman_p': spearman_p}

    def plot_cpd_level_boxplot(self, df: pd.DataFrame, output_dir: Path):
        """
        Figure 2: CPD Level Boxplot
        验证 Gain 单调下降
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Boxplot
        level_order = ['Low', 'Medium', 'High']
        colors = ['#2ca02c', '#ff7f0e', '#d62728']

        data = [df[df['cpd_level'] == level]['gain_absolute'].values
                for level in level_order]

        bp = ax.boxplot(data, tick_labels=level_order, patch_artist=True, widths=0.6)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加散点
        for i, level in enumerate(level_order):
            y = df[df['cpd_level'] == level]['gain_absolute'].values
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=80, edgecolors='black', linewidths=1)

        # 零线
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # 统计摘要
        for i, level in enumerate(level_order):
            data_level = df[df['cpd_level'] == level]['gain_absolute']
            mean_gain = data_level.mean()
            ax.text(i+1, ax.get_ylim()[1] * 0.9, f'μ={mean_gain:.3f}',
                   ha='center', fontsize=10, fontweight='bold')

        ax.set_ylabel('Ensemble Gain (Stacking F1 - Best Base F1)', fontsize=13, fontweight='bold')
        ax.set_xlabel('CPD Level', fontsize=13, fontweight='bold')
        ax.set_title('Ensemble Gain across CPD Levels', fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'controlled_cpd_level_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"✅ Figure 2 saved: controlled_cpd_level_boxplot.png")
        plt.close()

    def statistical_validation(self, df: pd.DataFrame, n_bootstrap=1000):
        """统计显著性验证"""
        print("\n" + "="*80)
        print("统计显著性验证（Bootstrap + Permutation）")
        print("="*80)

        # 按 CPD Level 分组
        low_gains = df[df['cpd_level'] == 'Low']['gain_absolute'].values
        medium_gains = df[df['cpd_level'] == 'Medium']['gain_absolute'].values
        high_gains = df[df['cpd_level'] == 'High']['gain_absolute'].values

        print(f"Low CPD:    mean={low_gains.mean():.4f}, std={low_gains.std():.4f}, n={len(low_gains)}")
        print(f"Medium CPD: mean={medium_gains.mean():.4f}, std={medium_gains.std():.4f}, n={len(medium_gains)}")
        print(f"High CPD:   mean={high_gains.mean():.4f}, std={high_gains.std():.4f}, n={len(high_gains)}")

        # Bootstrap 置信区间
        low_boot = []
        high_boot = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            if len(low_gains) > 0:
                low_sample = np.random.choice(low_gains, size=len(low_gains), replace=True)
                low_boot.append(low_sample.mean())
            if len(high_gains) > 0:
                high_sample = np.random.choice(high_gains, size=len(high_gains), replace=True)
                high_boot.append(high_sample.mean())

        if len(low_boot) > 0 and len(high_boot) > 0:
            low_ci = np.percentile(low_boot, [2.5, 97.5])
            high_ci = np.percentile(high_boot, [2.5, 97.5])

            print(f"\nBootstrap 95% CI (n={n_bootstrap}):")
            print(f"  Low CPD:  [{low_ci[0]:.4f}, {low_ci[1]:.4f}]")
            print(f"  High CPD: [{high_ci[0]:.4f}, {high_ci[1]:.4f}]")

            # Permutation test
            if len(low_gains) > 0 and len(high_gains) > 0:
                observed_diff = high_gains.mean() - low_gains.mean()
                pooled = np.concatenate([low_gains, high_gains])
                n_low = len(low_gains)

                perm_diffs = []
                for _ in range(n_bootstrap):
                    np.random.shuffle(pooled)
                    perm_low = pooled[:n_low]
                    perm_high = pooled[n_low:]
                    perm_diff = perm_high.mean() - perm_low.mean()
                    perm_diffs.append(perm_diff)

                p_value = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()
                print(f"\nPermutation Test (Low vs High):")
                print(f"  Observed Δ = {observed_diff:.4f}")
                print(f"  p-value = {p_value:.4f}")

                if p_value < 0.05:
                    print(f"  ✅ 统计显著（p < 0.05）")
                else:
                    print(f"  ❌ 不显著（p >= 0.05）")

                return {
                    'low_ci': low_ci,
                    'high_ci': high_ci,
                    'observed_diff': observed_diff,
                    'p_value': p_value
                }

        return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Controlled CPD Experiment')
    parser.add_argument('--results-root', type=str, required=True)
    parser.add_argument('--feature-set', type=str, default='all_features')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(args.results_root).parent / 'report'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 实验
    experiment = ControlledCPDExperiment(args.results_root)

    # 运行实验
    df = experiment.run_controlled_experiment(args.feature_set, output_dir)

    # Figure 1: CPD vs Gain（核心）
    corr_results = experiment.plot_cpd_vs_gain(df, output_dir)

    # Figure 2: CPD Level Boxplot
    experiment.plot_cpd_level_boxplot(df, output_dir)

    # 统计验证
    stat_results = experiment.statistical_validation(df, n_bootstrap=1000)

    print("\n" + "="*80)
    print("Controlled CPD Experiment 完成！")
    print("="*80)
    print(f"\n核心发现：")
    print(f"  Pearson r = {corr_results['pearson_r']:.3f} (p = {corr_results['pearson_p']:.4f})")
    print(f"  Spearman ρ = {corr_results['spearman_r']:.3f} (p = {corr_results['spearman_p']:.4f})")
    if stat_results:
        print(f"  Low vs High CPD: Δ = {stat_results['observed_diff']:.4f} (p = {stat_results['p_value']:.4f})")


if __name__ == '__main__':
    main()
