#!/usr/bin/env python3
"""
Figure 3: Confusion Topology Shift
展示低/中/高 CPD 场景下类别关系结构变化
"""

import pandas as pd
import numpy as np
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_cm(task, model='rf', fs='all_features', results_root='results/robust_v2/raw_all'):
    cm_path = Path(results_root) / task / fs / model / 'confusion_matrix.csv'
    if not cm_path.exists():
        return None
    return pd.read_csv(cm_path, index_col=0).values


def normalize_cm(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def plot_confusion_topology(task, ax, title, cpd_value, cpd_level, results_root):
    """绘制单个任务的 confusion topology graph"""
    cm = load_cm(task, 'rf', 'all_features', results_root)
    if cm is None:
        return

    cm_norm = normalize_cm(cm)
    class_names = ['Camera', 'Light_T1', 'Light_XM', 'Sensor', 'Socket']

    # 创建有向图
    G = nx.DiGraph()
    for cls in class_names:
        G.add_node(cls)

    # 添加边（只保留 > 5% 的混淆）
    threshold = 0.05
    for i, ci in enumerate(class_names):
        for j, cj in enumerate(class_names):
            if i != j and cm_norm[i, j] > threshold:
                G.add_edge(ci, cj, weight=cm_norm[i, j])

    # 布局
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # 节点（大小=召回率，颜色=召回率）
    node_sizes = [800 + cm_norm[i, i] * 1500 for i in range(len(class_names))]
    node_colors = [cm_norm[i, i] for i in range(len(class_names))]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                          cmap='RdYlGn', vmin=0, vmax=1, alpha=0.9, ax=ax)

    # 边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 6 for w in weights],
                          alpha=0.6, edge_color='gray', arrows=True, arrowsize=18, ax=ax,
                          connectionstyle='arc3,rad=0.1')

    # 边标签
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    # 节点标签
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    # 颜色映射（CPD Level）
    color_map = {'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}

    title_full = f'{title}\nCPD = {cpd_value:.3f} ({cpd_level})'
    ax.set_title(title_full, fontsize=10, fontweight='bold', color=color_map.get(cpd_level, 'black'))
    ax.axis('off')


def main():
    """Figure 3 主函数"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=str, default='results/robust_v2/raw_all')
    parser.add_argument('--data-csv', type=str, default='results/robust_v2/report/controlled_cpd_data.csv')
    parser.add_argument('--output-dir', type=str, default='results/robust_v2/report')
    args = parser.parse_args()

    # 加载数据
    df = pd.read_csv(args.data_csv)

    # 选择 6 个代表性任务（每个 CPD Level 2 个）
    selected = []
    for level in ['Low', 'Medium', 'High']:
        level_df = df[df['cpd_level'] == level].sort_values('cpd')
        if len(level_df) >= 2:
            selected.append(level_df.iloc[0])  # 最低 CPD
            selected.append(level_df.iloc[-1])  # 最高 CPD
        elif len(level_df) > 0:
            selected.append(level_df.iloc[0])

    # 创建 2x3 布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, row in enumerate(selected[:6]):
        task = row['task']
        title = task.replace('_R2_R3_R4_to_', '→').replace('single_round_', 'IID ').replace('loro_', 'LORO ').replace('position_', 'Position ').replace('jitter_', 'Jitter ').replace('joint_', 'Joint ')

        plot_confusion_topology(task, axes[idx], title, row['cpd'], row['cpd_level'], args.results_root)

    plt.suptitle('Figure 3: Confusion Topology Shift across CPD Levels\n(Node size ∝ Recall, Edge width ∝ Confusion Rate)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = Path(args.output_dir) / 'controlled_cpd_topology_shift.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3 saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
