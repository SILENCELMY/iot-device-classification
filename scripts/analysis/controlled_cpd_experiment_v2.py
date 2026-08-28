#!/usr/bin/env python3
"""
Controlled CPD Experiment V2 - 机制验证实验（改进版）
目标：通过控制 CPD 强度验证因果关系
核心假设：CPD ↑ => Ensemble Gain ↓

改进点：
1. 使用训练环境与测试环境之间的 pairwise CPD（基于 six_env 矩阵）
2. 更稳健的统计验证（bootstrap + permutation + 单调性检验）
3. 完整的 confusion topology graph 可视化
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
import json
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ControlledCPDExperimentV2:
    """Controlled CPD 机制验证实验 V2"""

    def __init__(self, results_root: str, six_env_cpd_path: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']
        self.n_classes = len(self.class_names)

        # 加载六环境 CPD 矩阵
        self.six_env_cpd = pd.read_csv(six_env_cpd_path, index_col=0)
        print(f"✓ 加载六环境 CPD 矩阵: {six_env_cpd_path}")
        print(self.six_env_cpd)

        # 基模型列表
        self.base_models = ['rf', 'xgboost', 'lightgbm', 'extra_trees']

        # 任务定义（训练环境 -> 测试环境）
        self.task_env_mapping = {
            'single_round_R2': (['R2'], ['R2']),
            'single_round_R3': (['R3'], ['R3']),
            'single_round_R4': (['R4'], ['R4']),
            'loro_R2_R3_to_R4': (['R2', 'R3'], ['R4']),
            'loro_R2_R4_to_R3': (['R2', 'R4'], ['R3']),
            'loro_R3_R4_to_R2': (['R3', 'R4'], ['R2']),
            'joint_R2_R3_R4': (['R2', 'R3', 'R4'], ['R2', 'R3', 'R4']),
            'position_R2_R3_R4_to_R5': (['R2', 'R3', 'R4'], ['R5']),
            'jitter_R2_R3_R4_to_R6': (['R2', 'R3', 'R4'], ['R6']),
            'jitter_R2_R3_R4_to_R7': (['R2', 'R3', 'R4'], ['R7']),
            'jitter_R2_R3_R4_to_R6_R7': (['R2', 'R3', 'R4'], ['R6', 'R7']),
        }

    def compute_task_cpd(self, train_envs: list, test_envs: list) -> float:
        """
        计算任务的 CPD：训练环境与测试环境之间的平均 pairwise CPD

        CPD_task = mean_{tr ∈ train, te ∈ test} CPD(tr, te)

        特殊情况：
        - 如果 train == test（IID），返回 0
        - 如果 train 包含 test（joint），返回训练环境内部的平均 CPD
        """
        # IID 场景
        if set(train_envs) == set(test_envs) and len(train_envs) == 1:
            return 0.0

        # Joint 场景（训练=测试=多环境）
        if set(train_envs) == set(test_envs) and len(train_envs) > 1:
            # 返回训练环境内部的平均 CPD（环境多样性）
            cpds = []
            for i, e1 in enumerate(train_envs):
                for e2 in train_envs[i+1:]:
                    cpds.append(self.six_env_cpd.loc[e1, e2])
            return np.mean(cpds) if cpds else 0.0

        # OOD 场景：训练环境 vs 测试环境
        cpds = []
        for tr_env in train_envs:
            for te_env in test_envs:
                if tr_env in self.six_env_cpd.index and te_env in self.six_env_cpd.columns:
                    cpds.append(self.six_env_cpd.loc[tr_env, te_env])

        return np.mean(cpds) if cpds else 0.0

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
        Gain = F1_stacking - max(F1_base_models)
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

        # Gain
        gain_absolute = stacking_f1 - best_base_f1
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
        """分类 CPD Level"""
        if cpd < 0.2:
            return 'Low'
        elif cpd < 0.4:
            return 'Medium'
        else:
            return 'High'

    def run_controlled_experiment(self, feature_set='all_features', output_dir=None):
        """运行 Controlled CPD 实验"""
        print("\n" + "="*80)
        print("Controlled CPD Experiment V2 - 机制验证")
        print("="*80)

        # 收集所有任务的数据
        experiment_data = []

        for task, (train_envs, test_envs) in self.task_env_mapping.items():
            # 计算 Ensemble Gain
            gain_info = self.compute_ensemble_gain(task, feature_set)
            if gain_info is None:
                print(f"⚠️  {task:40s} - 数据缺失")
                continue

            # 计算 CPD（基于环境映射）
            cpd = self.compute_task_cpd(train_envs, test_envs)
            cpd_level = self.classify_cpd_level(cpd)

            experiment_data.append({
                'task': task,
                'train_envs': '+'.join(train_envs),
                'test_envs': '+'.join(test_envs),
                'cpd': cpd,
                'cpd_level': cpd_level,
                'stacking_f1': gain_info['stacking_f1'],
                'best_base_f1': gain_info['best_base_f1'],
                'best_base_model': gain_info['best_base_model'],
                'gain_absolute': gain_info['gain_absolute'],
                'gain_relative': gain_info['gain_relative'],
            })

            print(f"{task:40s}  Train={train_envs} Test={test_envs}  "
                  f"CPD={cpd:.4f} ({cpd_level:6s})  Gain={gain_info['gain_absolute']:+.4f}")

        df = pd.DataFrame(experiment_data)

        # 保存数据
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(output_dir / 'controlled_cpd_data_v2.csv', index=False)
            print(f"\n✅ 数据已保存: controlled_cpd_data_v2.csv")

        return df

    def plot_cpd_vs_gain(self, df: pd.DataFrame, output_dir: Path):
        """
        Figure 1: CPD vs Ensemble Gain（最重要）
        验证核心假设：CPD ↑ => Gain ↓
        """
        fig, ax = plt.subplots(figsize=(11, 8))

        # 散点图（按 CPD Level 着色）
        colors = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
        markers = {'Low': 'o', 'Medium': 's', 'High': '^'}

        for level in ['Low', 'Medium', 'High']:
            mask = df['cpd_level'] == level
            if mask.sum() > 0:
                ax.scatter(df.loc[mask, 'cpd'],
                          df.loc[mask, 'gain_absolute'],
                          c=colors[level], marker=markers[level],
                          label=f'{level} CPD (n={mask.sum()})',
                          s=200, alpha=0.8, edgecolors='black', linewidths=1.5)

        # 拟合线
        if len(df) >= 3:
            z = np.polyfit(df['cpd'], df['gain_absolute'], 1)
            p = np.poly1d(z)
            x_fit = np.linspace(df['cpd'].min(), df['cpd'].max(), 100)
            ax.plot(x_fit, p(x_fit), '--', color='gray', alpha=0.7, linewidth=2.5,
                   label=f'Linear fit: y = {z[0]:.3f}x + {z[1]:.3f}')

        # 零线
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6,
                  label='Gain = 0 (No ensemble benefit)')

        # 相关性分析
        pearson_r, pearson_p = pearsonr(df['cpd'], df['gain_absolute'])
        spearman_r, spearman_p = spearmanr(df['cpd'], df['gain_absolute'])
        kendall_r, kendall_p = kendalltau(df['cpd'], df['gain_absolute'])

        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})\n'
        textstr += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})\n'
        textstr += f'Kendall τ = {kendall_r:.3f} (p={kendall_p:.4f})'

        # 标注显著性
        if pearson_p < 0.01:
            sig_text = '✅ Highly Significant (p < 0.01)'
        elif pearson_p < 0.05:
            sig_text = '✅ Significant (p < 0.05)'
        else:
            sig_text = '⚠️ Not Significant (p >= 0.05)'

        ax.text(0.05, 0.95, textstr + f'\n\n{sig_text}', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('CPD (Confusion Pattern Drift)\nTrain-Test Environment Distance',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Ensemble Gain (Stacking F1 - Best Base F1)',
                     fontsize=14, fontweight='bold')
        ax.set_title('Figure 1: Controlled CPD Experiment\nCPD vs Ensemble Gain',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / 'controlled_cpd_vs_gain_v2.png', dpi=300, bbox_inches='tight')
        print(f"✅ Figure 1 saved: controlled_cpd_vs_gain_v2.png")
        plt.close()

        return {
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p,
            'kendall_r': kendall_r, 'kendall_p': kendall_p
        }

    def plot_cpd_level_boxplot(self, df: pd.DataFrame, output_dir: Path):
        """
        Figure 2: CPD Level Boxplot
        验证 Gain 单调下降
        """
        fig, ax = plt.subplots(figsize=(11, 8))

        # Boxplot
        level_order = ['Low', 'Medium', 'High']
        colors = ['#2ca02c', '#ff7f0e', '#d62728']

        data = [df[df['cpd_level'] == level]['gain_absolute'].values
                for level in level_order]

        bp = ax.boxplot(data, tick_labels=level_order, patch_artist=True, widths=0.5)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 添加散点（jitter）
        for i, level in enumerate(level_order):
            y = df[df['cpd_level'] == level]['gain_absolute'].values
            x = np.random.normal(i+1, 0.05, size=len(y))
            ax.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidths=1.5, zorder=3)

        # 零线
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6,
                  label='Gain = 0')

        # 统计摘要
        for i, level in enumerate(level_order):
            data_level = df[df['cpd_level'] == level]['gain_absolute']
            if len(data_level) > 0:
                mean_gain = data_level.mean()
                n = len(data_level)
                ax.text(i+1, ax.get_ylim()[1] * 0.85,
                       f'μ={mean_gain:.4f}\nn={n}',
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel('Ensemble Gain (Stacking F1 - Best Base F1)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('CPD Level', fontsize=14, fontweight='bold')
        ax.set_title('Figure 2: Ensemble Gain across CPD Levels\n(Monotonic Decrease Hypothesis)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'controlled_cpd_level_boxplot_v2.png', dpi=300, bbox_inches='tight')
        print(f"✅ Figure 2 saved: controlled_cpd_level_boxplot_v2.png")
        plt.close()

    def statistical_validation(self, df: pd.DataFrame, n_bootstrap=10000):
        """统计显著性验证（增强版）"""
        print("\n" + "="*80)
        print("统计显著性验证（Bootstrap + Permutation + 单调性检验）")
        print("="*80)

        # 按 CPD Level 分组
        low_gains = df[df['cpd_level'] == 'Low']['gain_absolute'].values
        medium_gains = df[df['cpd_level'] == 'Medium']['gain_absolute'].values
        high_gains = df[df['cpd_level'] == 'High']['gain_absolute'].values

        print(f"\nCPD Level 分组统计:")
        print(f"  Low CPD:    mean={low_gains.mean():.4f}, std={low_gains.std():.4f}, n={len(low_gains)}")
        print(f"  Medium CPD: mean={medium_gains.mean():.4f}, std={medium_gains.std():.4f}, n={len(medium_gains)}")
        print(f"  High CPD:   mean={high_gains.mean():.4f}, std={high_gains.std():.4f}, n={len(high_gains)}")

        results = {}

        # 1. Bootstrap 置信区间
        print(f"\n1. Bootstrap 95% CI (n={n_bootstrap}):")
        np.random.seed(42)

        for level_name, level_data in [('Low', low_gains), ('Medium', medium_gains), ('High', high_gains)]:
            if len(level_data) > 0:
                boot_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(level_data, size=len(level_data), replace=True)
                    boot_means.append(sample.mean())
                ci = np.percentile(boot_means, [2.5, 97.5])
                print(f"  {level_name:6s} CPD: [{ci[0]:+.4f}, {ci[1]:+.4f}]")
                results[f'{level_name.lower()}_ci'] = ci

        # 2. Permutation Test (Low vs High)
        if len(low_gains) > 0 and len(high_gains) > 0:
            print(f"\n2. Permutation Test (Low vs High):")
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
            print(f"  Observed Δ(High - Low) = {observed_diff:.4f}")
            print(f"  p-value = {p_value:.4f}")

            if p_value < 0.01:
                print(f"  ✅✅ 高度显著（p < 0.01）")
            elif p_value < 0.05:
                print(f"  ✅ 显著（p < 0.05）")
            else:
                print(f"  ⚠️ 不显著（p >= 0.05）")

            results['perm_observed_diff'] = observed_diff
            results['perm_p_value'] = p_value

        # 3. 单调性检验（Jonckheere-Terpstra trend test 的简化版）
        print(f"\n3. 单调趋势检验:")
        level_means = []
        for level_name, level_data in [('Low', low_gains), ('Medium', medium_gains), ('High', high_gains)]:
            if len(level_data) > 0:
                level_means.append(level_data.mean())

        if len(level_means) == 3:
            is_monotonic = level_means[0] > level_means[1] > level_means[2]
            print(f"  Low→Medium→High 均值: {level_means[0]:.4f} → {level_means[1]:.4f} → {level_means[2]:.4f}")
            print(f"  单调递减: {'✅ 是' if is_monotonic else '❌ 否'}")
            results['monotonic_decrease'] = is_monotonic

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Controlled CPD Experiment V2')
    parser.add_argument('--results-root', type=str,
                       default='results/robust_v2/raw_all')
    parser.add_argument('--six-env-cpd', type=str,
                       default='results/robust_v2/report/six_env_off_diag_frobenius_rf.csv')
    parser.add_argument('--feature-set', type=str, default='all_features')
    parser.add_argument('--output-dir', type=str,
                       default='results/robust_v2/report')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 实验
    experiment = ControlledCPDExperimentV2(args.results_root, args.six_env_cpd)

    # 运行实验
    df = experiment.run_controlled_experiment(args.feature_set, output_dir)

    # Figure 1: CPD vs Gain（核心）
    corr_results = experiment.plot_cpd_vs_gain(df, output_dir)

    # Figure 2: CPD Level Boxplot
    experiment.plot_cpd_level_boxplot(df, output_dir)

    # 统计验证
    stat_results = experiment.statistical_validation(df, n_bootstrap=10000)

    print("\n" + "="*80)
    print("Controlled CPD Experiment V2 完成！")
    print("="*80)
    print(f"\n核心发现：")
    print(f"  Pearson r = {corr_results['pearson_r']:.3f} (p = {corr_results['pearson_p']:.4f})")
    print(f"  Spearman ρ = {corr_results['spearman_r']:.3f} (p = {corr_results['spearman_p']:.4f})")
    print(f"  Kendall τ = {corr_results['kendall_r']:.3f} (p = {corr_results['kendall_p']:.4f})")
    if 'perm_p_value' in stat_results:
        print(f"  Permutation Test: Δ = {stat_results['perm_observed_diff']:.4f} (p = {stat_results['perm_p_value']:.4f})")
    if 'monotonic_decrease' in stat_results:
        print(f"  单调递减: {stat_results['monotonic_decrease']}")


if __name__ == '__main__':
    main()
