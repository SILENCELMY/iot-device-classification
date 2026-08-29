#!/usr/bin/env python3
"""
CPD (Confusion Pattern Drift) 综合分析
论文核心：Class Relationship Structure Drift 导致 OOD Ensemble Failure

包含：
1. CPD vs OOD Performance Correlation (Task 3)
2. Confusion Topology Graph Visualization (Task 4)
3. Statistical Significance Tests (Task 7)
4. Paper-level Analysis Report (Task 8)
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
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CPDComprehensiveAnalyzer:
    """CPD 综合分析器"""

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']
        self.n_classes = len(self.class_names)

        # 环境映射
        self.env_mapping = {
            'R2': 'single_round_R2',
            'R3': 'single_round_R3',
            'R4': 'single_round_R4',
            'R5': 'position_R2_R3_R4_to_R5',
            'R6': 'jitter_R2_R3_R4_to_R6',
            'R7': 'jitter_R2_R3_R4_to_R7',
        }

        # 所有实验任务（扩展版）
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
        ]

    def compute_cpd(self, cm1: np.ndarray, cm2: np.ndarray, normalize=True) -> float:
        """
        计算 CPD (Confusion Pattern Drift)
        CPD(e_i, e_j) = ||Off(C_i) - Off(C_j)||_F

        Args:
            cm1, cm2: confusion matrices
            normalize: 是否先归一化为行和为1
        """
        if normalize:
            cm1 = self._normalize_cm(cm1)
            cm2 = self._normalize_cm(cm2)

        # 去除对角线
        off1 = cm1.copy()
        off2 = cm2.copy()
        np.fill_diagonal(off1, 0)
        np.fill_diagonal(off2, 0)

        # Frobenius distance
        return np.linalg.norm(off1 - off2, ord='fro')

    def _normalize_cm(self, cm: np.ndarray) -> np.ndarray:
        """行归一化混淆矩阵"""
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

    def load_f1(self, task: str, model: str = 'rf', feature_set: str = 'all_features') -> float:
        """加载 macro-F1"""
        metrics_path = self.results_root / task / feature_set / model / 'metrics.json'
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            metrics = json.load(f)
        return metrics.get('macro_f1')

    # ============================================================
    # Task 3: CPD vs OOD Performance Correlation
    # ============================================================

    def analyze_cpd_performance_correlation(self, model='rf', feature_set='all_features',
                                           output_dir=None):
        """
        分析 CPD 与 OOD 性能下降的相关性
        验证假设：CPD 越大 → 性能下降越严重
        """
        print("\n" + "="*80)
        print("Task 3: CPD vs OOD Performance Correlation Analysis")
        print("="*80)

        # 基准：IID 平均性能
        iid_tasks = ['single_round_R2', 'single_round_R3', 'single_round_R4']
        iid_f1s = [self.load_f1(t, model, feature_set) for t in iid_tasks]
        iid_f1s = [f for f in iid_f1s if f is not None]
        iid_baseline = np.mean(iid_f1s)
        print(f"IID Baseline (avg): {iid_baseline:.4f}")

        # OOD 任务
        ood_tasks = [
            'loro_R2_R3_to_R4',
            'loro_R2_R4_to_R3',
            'loro_R3_R4_to_R2',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6',
            'jitter_R2_R3_R4_to_R7',
        ]

        # 收集数据
        cpd_scores = []
        performance_drops = []
        task_labels = []

        for ood_task in ood_tasks:
            ood_f1 = self.load_f1(ood_task, model, feature_set)
            if ood_f1 is None:
                continue

            # 性能下降
            perf_drop = iid_baseline - ood_f1

            # CPD: 与 IID baseline 的平均 CPD
            ood_cm = self.load_cm(ood_task, model, feature_set)
            if ood_cm is None:
                continue

            iid_cpds = []
            for iid_task in iid_tasks:
                iid_cm = self.load_cm(iid_task, model, feature_set)
                if iid_cm is not None:
                    cpd = self.compute_cpd(iid_cm, ood_cm)
                    iid_cpds.append(cpd)

            if len(iid_cpds) > 0:
                avg_cpd = np.mean(iid_cpds)
                cpd_scores.append(avg_cpd)
                performance_drops.append(perf_drop)
                task_labels.append(ood_task)

                print(f"  {ood_task:40s}  CPD={avg_cpd:.4f}  ΔF1={perf_drop:+.4f}")

        # 统计相关性
        if len(cpd_scores) >= 3:
            pearson_r, pearson_p = pearsonr(cpd_scores, performance_drops)
            spearman_r, spearman_p = spearmanr(cpd_scores, performance_drops)

            print(f"\n统计相关性:")
            print(f"  Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
            print(f"  Spearman: ρ={spearman_r:.4f}, p={spearman_p:.4f}")

            # 可视化
            if output_dir:
                self._plot_cpd_performance_correlation(
                    cpd_scores, performance_drops, task_labels,
                    pearson_r, pearson_p, spearman_r, spearman_p,
                    output_dir
                )

            return {
                'cpd_scores': cpd_scores,
                'performance_drops': performance_drops,
                'task_labels': task_labels,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
            }
        else:
            print("数据点不足，无法进行相关性分析")
            return None

    def _plot_cpd_performance_correlation(self, cpd_scores, performance_drops, task_labels,
                                         pearson_r, pearson_p, spearman_r, spearman_p,
                                         output_dir):
        """绘制 CPD vs Performance Drop 散点图"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # 散点图
        colors = ['#d62728' if 'loro' in t else '#ff7f0e' if 'position' in t else '#2ca02c'
                 for t in task_labels]

        ax.scatter(cpd_scores, performance_drops, s=150, alpha=0.7,
                  c=colors, edgecolors='black', linewidths=1.5)

        # 标注任务名
        for i, label in enumerate(task_labels):
            # 简化标签
            short_label = label.replace('_R2_R3_R4_to_', '→').replace('loro_', 'LORO ')
            short_label = short_label.replace('position_', 'Position ').replace('jitter_', 'Jitter ')
            ax.annotate(short_label, (cpd_scores[i], performance_drops[i]),
                       fontsize=9, ha='left', xytext=(5, 5), textcoords='offset points')

        # 拟合线
        if len(cpd_scores) >= 2:
            z = np.polyfit(cpd_scores, performance_drops, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(min(cpd_scores), max(cpd_scores), 100)
            ax.plot(x_fit, p(x_fit), '--', color='gray', alpha=0.7, linewidth=2)

        # 相关系数注释
        textstr = f'Pearson r = {pearson_r:.3f} (p={pearson_p:.4f})\n'
        textstr += f'Spearman ρ = {spearman_r:.3f} (p={spearman_p:.4f})'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('CPD (Confusion Pattern Drift)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance Drop (IID Baseline - OOD F1)', fontsize=13, fontweight='bold')
        ax.set_title('CPD vs OOD Performance Degradation', fontsize=15, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / 'cpd_vs_performance_correlation.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ CPD correlation plot saved")
        plt.close()

    # ============================================================
    # Task 4: Confusion Topology Graph Visualization
    # ============================================================

    def visualize_confusion_topology(self, tasks, model='rf', feature_set='all_features',
                                    output_dir=None):
        """
        将 confusion matrix 视为有向图
        node = class
        edge = P(ŷ=j | y=i) (confusion probability)
        """
        print("\n" + "="*80)
        print("Task 4: Confusion Topology Graph Visualization")
        print("="*80)

        n_tasks = len(tasks)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, task in enumerate(tasks):
            cm = self.load_cm(task, model, feature_set)
            if cm is None:
                continue

            cm_norm = self._normalize_cm(cm)

            # 创建有向图
            G = nx.DiGraph()
            for i, class_i in enumerate(self.class_names):
                G.add_node(class_i)

            # 添加边（只保留 confusion，不包括对角线）
            threshold = 0.05  # 只显示 > 5% 的混淆
            for i, class_i in enumerate(self.class_names):
                for j, class_j in enumerate(self.class_names):
                    if i != j and cm_norm[i, j] > threshold:
                        G.add_edge(class_i, class_j, weight=cm_norm[i, j])

            # 绘制
            ax = axes[idx]
            pos = nx.spring_layout(G, seed=42, k=2)

            # 绘制节点
            node_sizes = [1000 + cm_norm[i, i] * 2000 for i in range(self.n_classes)]
            node_colors = [cm_norm[i, i] for i in range(self.n_classes)]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                  node_color=node_colors, cmap='RdYlGn',
                                  vmin=0, vmax=1, alpha=0.9, ax=ax)

            # 绘制边
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*5 for w in weights],
                                  alpha=0.6, edge_color='gray',
                                  arrows=True, arrowsize=20, ax=ax,
                                  connectionstyle='arc3,rad=0.1')

            # 边标签
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

            # 节点标签
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

            # 标题
            task_short = task.replace('_R2_R3_R4_to_', '→').replace('single_round_', 'IID ')
            task_short = task_short.replace('loro_', 'LORO ').replace('position_', 'Position ')
            task_short = task_short.replace('jitter_', 'Jitter ')
            f1 = self.load_f1(task, model, feature_set)
            title = f'{task_short}\nF1={f1:.3f}' if f1 else task_short
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

        plt.suptitle('Confusion Topology Graphs (Node size ∝ Recall, Edge width ∝ Confusion Rate)',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        if output_dir:
            plt.savefig(output_dir / 'confusion_topology_graphs.png', dpi=300, bbox_inches='tight')
            print(f"✅ Confusion topology graphs saved")
        plt.close()

    # ============================================================
    # Task 7: Statistical Significance Tests
    # ============================================================

    def statistical_significance_test(self, model='rf', feature_set='all_features',
                                     n_bootstrap=1000, output_dir=None):
        """
        Bootstrap + Permutation Test
        验证 IID vs OOD 的 CPD 差异是否统计显著
        """
        print("\n" + "="*80)
        print("Task 7: Statistical Significance Tests")
        print("="*80)

        iid_tasks = ['single_round_R2', 'single_round_R3', 'single_round_R4']
        ood_tasks = [
            'loro_R2_R3_to_R4',
            'loro_R2_R4_to_R3',
            'loro_R3_R4_to_R2',
            'position_R2_R3_R4_to_R5',
            'jitter_R2_R3_R4_to_R6',
            'jitter_R2_R3_R4_to_R7',
        ]

        # 计算 IID 内部 CPD
        iid_cpds = []
        for i in range(len(iid_tasks)):
            for j in range(i+1, len(iid_tasks)):
                cm_i = self.load_cm(iid_tasks[i], model, feature_set)
                cm_j = self.load_cm(iid_tasks[j], model, feature_set)
                if cm_i is not None and cm_j is not None:
                    cpd = self.compute_cpd(cm_i, cm_j)
                    iid_cpds.append(cpd)

        # 计算 OOD vs IID CPD
        ood_cpds = []
        for ood_task in ood_tasks:
            ood_cm = self.load_cm(ood_task, model, feature_set)
            if ood_cm is None:
                continue
            for iid_task in iid_tasks:
                iid_cm = self.load_cm(iid_task, model, feature_set)
                if iid_cm is not None:
                    cpd = self.compute_cpd(iid_cm, ood_cm)
                    ood_cpds.append(cpd)

        iid_cpds = np.array(iid_cpds)
        ood_cpds = np.array(ood_cpds)

        print(f"IID 内部 CPD:  mean={iid_cpds.mean():.4f}, std={iid_cpds.std():.4f}, n={len(iid_cpds)}")
        print(f"OOD vs IID CPD: mean={ood_cpds.mean():.4f}, std={ood_cpds.std():.4f}, n={len(ood_cpds)}")

        # Bootstrap 置信区间
        iid_boot_means = []
        ood_boot_means = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            iid_sample = np.random.choice(iid_cpds, size=len(iid_cpds), replace=True)
            ood_sample = np.random.choice(ood_cpds, size=len(ood_cpds), replace=True)
            iid_boot_means.append(iid_sample.mean())
            ood_boot_means.append(ood_sample.mean())

        iid_ci = np.percentile(iid_boot_means, [2.5, 97.5])
        ood_ci = np.percentile(ood_boot_means, [2.5, 97.5])

        print(f"\nBootstrap 95% CI (n={n_bootstrap}):")
        print(f"  IID:  [{iid_ci[0]:.4f}, {iid_ci[1]:.4f}]")
        print(f"  OOD:  [{ood_ci[0]:.4f}, {ood_ci[1]:.4f}]")

        # Permutation test
        observed_diff = ood_cpds.mean() - iid_cpds.mean()
        pooled = np.concatenate([iid_cpds, ood_cpds])
        n_iid = len(iid_cpds)

        perm_diffs = []
        for _ in range(n_bootstrap):
            np.random.shuffle(pooled)
            perm_iid = pooled[:n_iid]
            perm_ood = pooled[n_iid:]
            perm_diff = perm_ood.mean() - perm_iid.mean()
            perm_diffs.append(perm_diff)

        p_value = (np.abs(perm_diffs) >= np.abs(observed_diff)).mean()
        print(f"\nPermutation Test:")
        print(f"  Observed Δ = {observed_diff:.4f}")
        print(f"  p-value = {p_value:.4f}")

        if p_value < 0.05:
            print(f"  ✅ 统计显著（p < 0.05）")
        else:
            print(f"  ❌ 不显著（p >= 0.05）")

        # 可视化
        if output_dir:
            self._plot_significance_test(iid_cpds, ood_cpds, iid_boot_means, ood_boot_means,
                                        perm_diffs, observed_diff, p_value, output_dir)

        return {
            'iid_cpds': iid_cpds,
            'ood_cpds': ood_cpds,
            'iid_ci': iid_ci,
            'ood_ci': ood_ci,
            'observed_diff': observed_diff,
            'p_value': p_value,
        }

    def _plot_significance_test(self, iid_cpds, ood_cpds, iid_boot_means, ood_boot_means,
                               perm_diffs, observed_diff, p_value, output_dir):
        """绘制统计显著性检验可视化"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Box plot
        ax1 = axes[0]
        data = [iid_cpds, ood_cpds]
        bp = ax1.boxplot(data, tick_labels=['IID 内部', 'OOD vs IID'],
                        patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2ca02c')
        bp['boxes'][1].set_facecolor('#d62728')
        ax1.set_ylabel('CPD (Off-diagonal Frobenius Distance)', fontsize=11, fontweight='bold')
        ax1.set_title('CPD Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Bootstrap distributions
        ax2 = axes[1]
        ax2.hist(iid_boot_means, bins=30, alpha=0.6, color='#2ca02c', label='IID', density=True)
        ax2.hist(ood_boot_means, bins=30, alpha=0.6, color='#d62728', label='OOD', density=True)
        ax2.axvline(np.mean(iid_cpds), color='darkgreen', linestyle='--', linewidth=2, label='IID mean')
        ax2.axvline(np.mean(ood_cpds), color='darkred', linestyle='--', linewidth=2, label='OOD mean')
        ax2.set_xlabel('Bootstrap Mean CPD', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax2.set_title('Bootstrap Distributions (n=1000)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Permutation test
        ax3 = axes[2]
        ax3.hist(perm_diffs, bins=40, alpha=0.7, color='gray', edgecolor='black', density=True)
        ax3.axvline(observed_diff, color='red', linestyle='--', linewidth=3,
                   label=f'Observed Δ = {observed_diff:.4f}')
        ax3.set_xlabel('Permuted Δ (OOD - IID)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax3.set_title(f'Permutation Test (p={p_value:.4f})', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'cpd_statistical_significance.png', dpi=300, bbox_inches='tight')
        print(f"✅ Statistical significance plot saved")
        plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CPD Comprehensive Analysis')
    parser.add_argument('--results-root', type=str, required=True)
    parser.add_argument('--model', type=str, default='rf')
    parser.add_argument('--feature-set', type=str, default='all_features')
    parser.add_argument('--output-dir', type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path(args.results_root).parent / 'report'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    analyzer = CPDComprehensiveAnalyzer(args.results_root)

    print("="*80)
    print("CPD (Confusion Pattern Drift) 综合分析")
    print("="*80)

    # Task 3: CPD vs Performance Correlation
    corr_results = analyzer.analyze_cpd_performance_correlation(
        args.model, args.feature_set, output_dir
    )

    # Task 4: Confusion Topology Graph
    topology_tasks = [
        'single_round_R2',
        'single_round_R3',
        'loro_R2_R4_to_R3',
        'position_R2_R3_R4_to_R5',
        'jitter_R2_R3_R4_to_R6',
        'jitter_R2_R3_R4_to_R7',
    ]
    analyzer.visualize_confusion_topology(topology_tasks, args.model, args.feature_set, output_dir)

    # Task 7: Statistical Significance
    sig_results = analyzer.statistical_significance_test(
        args.model, args.feature_set, n_bootstrap=1000, output_dir=output_dir
    )

    print("\n" + "="*80)
    print("CPD 综合分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
