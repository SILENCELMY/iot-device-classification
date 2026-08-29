#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的真实数据分析流程
从真实pcap数据到最终评估报告

包括：
1. 真实特征稳定性分析（从R2-R7真实数据）
2. 真实稳定性感知特征选择
3. 真实性能评估
"""

import sys
from pathlib import Path

# 添加code/scripts到Python路径
code_dir = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(code_dir))

from robust_iot_research import build_feature_table, META_COLUMNS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from typing import Dict, List, Tuple

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RealDataPipeline:
    """完整的真实数据分析流程"""

    def __init__(self, project_root: str, output_root: str):
        self.project_root = Path(project_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # 加载配置
        config_path = self.project_root / "code" / "configs" / "research_experiments.json"
        with open(config_path) as f:
            self.config = json.load(f)

        self.labels = self.config['labels']
        self.normal_rounds = ['R2', 'R3', 'R4']
        self.position_rounds = ['R5']
        self.jitter_rounds = ['R6', 'R7']
        self.all_rounds = self.normal_rounds + self.position_rounds + self.jitter_rounds

    def step1_extract_features(self) -> pd.DataFrame:
        """步骤1: 从真实pcap提取特征"""
        print("\n" + "="*70)
        print("步骤1: 从真实pcap文件提取特征".center(70))
        print("="*70)

        dataset_root = self.project_root / "dataset"
        cache_dir = self.output_root / "feature_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        features = build_feature_table(
            config=self.config,
            dataset_root=dataset_root,
            output_dir=cache_dir,
            required=set(self.all_rounds),
            filter_mode='raw_all',
            window_seconds=10.0,
            min_packets_per_window=2,
            force_extract=False
        )

        print(f"\n[INFO] 提取完成: {len(features)} 个样本")
        print(f"[INFO] 特征数量: {len([c for c in features.columns if c not in META_COLUMNS])} 个")

        return features

    def step2_analyze_stability(self, features: pd.DataFrame) -> pd.DataFrame:
        """步骤2: 真实特征稳定性分析"""
        print("\n" + "="*70)
        print("步骤2: 计算真实特征稳定性指标".center(70))
        print("="*70)

        # 获取特征列（排除元数据列）
        feature_cols = [c for c in features.columns if c not in META_COLUMNS]
        print(f"\n[INFO] 分析 {len(feature_cols)} 个特征的稳定性")

        stability_metrics = []

        for feature in feature_cols:
            # 计算各轮次的特征均值
            round_means = {}
            for round_name in self.all_rounds:
                round_data = features[features['round'] == round_name][feature]
                if len(round_data) > 0:
                    round_means[round_name] = round_data.mean()
                else:
                    round_means[round_name] = 0

            # 计算稳定性指标
            all_means = list(round_means.values())

            # 1. 总体统计
            mean_value = np.mean(all_means)
            std_value = np.std(all_means)
            cv = std_value / mean_value if mean_value > 1e-10 else 0

            # 2. 场景间漂移
            normal_mean = np.mean([round_means[r] for r in self.normal_rounds])
            position_mean = np.mean([round_means[r] for r in self.position_rounds])
            jitter_mean = np.mean([round_means[r] for r in self.jitter_rounds])

            shift_position = abs(normal_mean - position_mean) / (abs(normal_mean) + 1e-10)
            shift_jitter = abs(normal_mean - jitter_mean) / (abs(normal_mean) + 1e-10)
            shift_score = (shift_position + shift_jitter) / 2

            # 综合稳定性得分（越低越稳定）
            stability_score = 0.5 * cv + 0.5 * shift_score

            stability_metrics.append({
                'feature': feature,
                'mean': mean_value,
                'std': std_value,
                'cv': cv,
                'shift_position': shift_position,
                'shift_jitter': shift_jitter,
                'shift_score': shift_score,
                'stability_score': stability_score
            })

        stability_df = pd.DataFrame(stability_metrics).sort_values('stability_score')

        # 保存稳定性指标
        stability_path = self.output_root / 'real_feature_stability.csv'
        stability_df.to_csv(stability_path, index=False)
        print(f"\n[INFO] 稳定性指标已保存: {stability_path}")

        # 显示Top10最稳定和最不稳定的特征
        print(f"\n最稳定的10个特征:")
        for idx, row in stability_df.head(10).iterrows():
            print(f"  {row['feature']}: stability_score={row['stability_score']:.4f}, cv={row['cv']:.4f}")

        print(f"\n最不稳定的10个特征:")
        for idx, row in stability_df.tail(10).iterrows():
            print(f"  {row['feature']}: stability_score={row['stability_score']:.4f}, cv={row['cv']:.4f}")

        return stability_df

    def step3_select_features(self, features: pd.DataFrame, stability_df: pd.DataFrame) -> Dict[int, List[str]]:
        """步骤3: 基于真实稳定性选择特征"""
        print("\n" + "="*70)
        print("步骤3: 稳定性感知特征选择".center(70))
        print("="*70)

        # 读取特征重要性（从已有的feature_rankings）
        importance_path = self.project_root / "results" / "cloud_smoke" / "feature_rankings_all_tasks.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            importance_df = importance_df[importance_df['task'] == 'single_round_R2'][
                ['feature', 'mutual_info', 'model_importance']
            ]
        else:
            print("[WARNING] 未找到特征重要性文件，使用均匀权重")
            importance_df = pd.DataFrame({
                'feature': stability_df['feature'],
                'mutual_info': 1.0,
                'model_importance': 1.0
            })

        # 合并稳定性和重要性
        combined = pd.merge(stability_df, importance_df, on='feature', how='inner')

        # 归一化
        combined['importance_norm'] = (combined['mutual_info'] - combined['mutual_info'].min()) / \
                                     (combined['mutual_info'].max() - combined['mutual_info'].min() + 1e-10)

        combined['stability_norm'] = (combined['stability_score'] - combined['stability_score'].min()) / \
                                    (combined['stability_score'].max() - combined['stability_score'].min() + 1e-10)
        combined['stability_weight'] = 1 - combined['stability_norm']  # 反转：越稳定权重越高

        # 综合评分
        combined['combined_score'] = 0.6 * combined['importance_norm'] + 0.4 * combined['stability_weight']
        combined = combined.sort_values('combined_score', ascending=False)

        # 选择Top K
        selected_features = {}
        for k in [10, 15, 20]:
            selected_features[k] = combined.head(k)['feature'].tolist()
            print(f"\nTop{k}特征:")
            print(", ".join(selected_features[k][:5]) + ", ...")

        # 保存
        scores_path = self.output_root / 'feature_combined_scores.csv'
        combined.to_csv(scores_path, index=False)
        print(f"\n[INFO] 综合评分已保存: {scores_path}")

        for k, feats in selected_features.items():
            feat_path = self.output_root / f'selected_features_top{k}.txt'
            with open(feat_path, 'w') as f:
                f.write('\n'.join(feats))
            print(f"[INFO] Top{k}特征已保存: {feat_path}")

        return selected_features

    def step4_evaluate(self, features: pd.DataFrame, selected_features: Dict[int, List[str]]):
        """步骤4: 真实性能评估"""
        print("\n" + "="*70)
        print("步骤4: 真实性能评估".center(70))
        print("="*70)

        all_results = {}

        for top_k, feature_list in selected_features.items():
            print(f"\n{'='*70}")
            print(f"评估 Top{top_k} 特征".center(70))
            print(f"{'='*70}")

            results = self._evaluate_feature_set(features, feature_list, f"Top{top_k}")
            all_results[top_k] = results

            # 绘制混淆矩阵
            self._plot_confusion_matrices(results, top_k)

        # 生成最终报告
        self._generate_final_report(all_results, selected_features)

    def _evaluate_feature_set(self, features: pd.DataFrame, feature_list: List[str], name: str) -> List[Dict]:
        """评估一个特征集"""
        available_features = [f for f in feature_list if f in features.columns]
        print(f"\n[INFO] 使用 {len(available_features)}/{len(feature_list)} 个特征")

        scenarios = [
            ('Joint_R2R3R4', ['R2', 'R3', 'R4'], ['R2', 'R3', 'R4'], True),
            ('LORO_R2R3_to_R4', ['R2', 'R3'], ['R4'], False),
            ('LORO_R2R4_to_R3', ['R2', 'R4'], ['R3'], False),
            ('LORO_R3R4_to_R2', ['R3', 'R4'], ['R2'], False),
            ('Position_to_R5', ['R2', 'R3', 'R4'], ['R5'], False),
            ('Jitter_to_R6', ['R2', 'R3', 'R4'], ['R6'], False),
            ('Jitter_to_R7', ['R2', 'R3', 'R4'], ['R7'], False),
        ]

        results = []

        for scenario_name, train_rounds, test_rounds, use_split in scenarios:
            if use_split:
                # 内部验证：使用train_test_split
                data = features[features['round'].isin(train_rounds)]
                X = data[available_features].values
                y = data['label'].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            else:
                # 固定分割
                train_data = features[features['round'].isin(train_rounds)]
                test_data = features[features['round'].isin(test_rounds)]
                X_train = train_data[available_features].values
                X_test = test_data[available_features].values
                y_train = train_data['label'].values
                y_test = test_data['label'].values

            # 训练
            rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)

            # 预测
            y_pred = rf.predict(X_test)

            # 指标
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='macro', zero_division=0
            )
            _, _, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, zero_division=0, labels=self.labels
            )
            cm = confusion_matrix(y_test, y_pred, labels=self.labels)

            print(f"  {scenario_name}: F1={f1:.4f}, Acc={accuracy:.4f}")

            results.append({
                'scenario': f"{name}_{scenario_name}",
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'macro_f1': f1,
                'f1_per_class': dict(zip(self.labels, f1_per_class)),
                'confusion_matrix': cm,
                'support': len(y_test)
            })

        return results

    def _plot_confusion_matrices(self, results: List[Dict], top_k: int):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, result in enumerate(results):
            cm = result['confusion_matrix']
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

            sns.heatmap(
                cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=self.labels, yticklabels=self.labels,
                ax=axes[idx], cbar=True, vmin=0, vmax=1
            )

            title = result['scenario'].replace(f'Top{top_k}_', '')
            axes[idx].set_title(title, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')

        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        output_path = self.output_root / f'confusion_matrices_top{top_k}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 混淆矩阵已保存: {output_path}")
        plt.close()

    def _generate_final_report(self, all_results: Dict, selected_features: Dict):
        """生成最终报告"""
        print("\n[INFO] 生成最终报告...")

        lines = []
        lines.append("# 稳定性感知特征选择 - 完整真实数据分析报告\n")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("**数据来源**: 真实pcap文件（R2-R7）\n")
        lines.append("**重要说明**: 本报告所有数据均来自真实提取和训练，无任何模拟数据！\n")
        lines.append("---\n")

        lines.append("\n## 1. 评估设置\n")
        lines.append("- **设备类别**: " + ", ".join(self.labels) + "\n")
        lines.append("- **轮次**: R2-R4（正常），R5（位置变化），R6-R7（操作抖动）\n")
        lines.append("- **模型**: RandomForest (n_estimators=100)\n")
        lines.append("- **评估场景**: 联合训练、3个LORO、3个跨场景泛化\n")

        for top_k in [10, 15, 20]:
            if top_k not in all_results:
                continue

            results = all_results[top_k]
            lines.append(f"\n## 2.{top_k//5} Top{top_k}特征性能\n")

            # 性能表
            lines.append("\n### Macro-F1 性能表\n")
            lines.append("| 场景 | Macro-F1 | Accuracy | Precision | Recall | 样本数 |\n")
            lines.append("|------|----------|----------|-----------|--------|--------|\n")

            for r in results:
                name = r['scenario'].replace(f'Top{top_k}_', '')
                lines.append(f"| {name} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
                           f"{r['precision']:.4f} | {r['recall']:.4f} | {r['support']} |\n")

            # 每类F1
            lines.append("\n### 各设备F1分数\n")
            lines.append("| 场景 | " + " | ".join(self.labels) + " |\n")
            lines.append("|------|" + "|".join(["---------"]*len(self.labels)) + "|\n")

            for r in results:
                name = r['scenario'].replace(f'Top{top_k}_', '')
                f1_scores = [f"{r['f1_per_class'].get(l, 0):.4f}" for l in self.labels]
                lines.append(f"| {name} | " + " | ".join(f1_scores) + " |\n")

            lines.append(f"\n### 混淆矩阵\n")
            lines.append(f"![混淆矩阵](./confusion_matrices_top{top_k}.png)\n")

            # 关键指标
            joint_f1 = next(r['macro_f1'] for r in results if 'Joint' in r['scenario'])
            r5_f1 = next(r['macro_f1'] for r in results if 'R5' in r['scenario'])
            r6_f1 = next(r['macro_f1'] for r in results if 'R6' in r['scenario'])
            r7_f1 = next(r['macro_f1'] for r in results if 'R7' in r['scenario'])
            avg_cross = (r5_f1 + r6_f1 + r7_f1) / 3

            lines.append(f"\n### 关键指标\n")
            lines.append(f"- 联合训练: **{joint_f1:.4f}**\n")
            lines.append(f"- R5（位置变化）: **{r5_f1:.4f}**\n")
            lines.append(f"- R6（操作抖动）: **{r6_f1:.4f}**\n")
            lines.append(f"- R7（操作抖动）: **{r7_f1:.4f}**\n")
            lines.append(f"- **平均跨场景F1**: **{avg_cross:.4f}**\n")

        report_path = self.output_root / 'REAL_DATA_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"[INFO] 最终报告已保存: {report_path}")


def main():
    print("\n" + "="*70)
    print("完整真实数据分析流程".center(70))
    print("="*70)
    print("\n⚠️  重要说明：本流程使用100%真实数据，无任何模拟！\n")

    project_root = Path(__file__).parent
    output_root = project_root / "results" / "cloud_smoke" / "REAL_ANALYSIS"

    pipeline = RealDataPipeline(str(project_root), str(output_root))

    # 步骤1: 提取真实特征
    features = pipeline.step1_extract_features()

    # 步骤2: 真实稳定性分析
    stability_df = pipeline.step2_analyze_stability(features)

    # 步骤3: 真实特征选择
    selected_features = pipeline.step3_select_features(features, stability_df)

    # 步骤4: 真实评估
    pipeline.step4_evaluate(features, selected_features)

    print("\n" + "="*70)
    print("完整流程完成！".center(70))
    print("="*70)
    print(f"\n所有结果保存在: {output_root}\n")


if __name__ == "__main__":
    main()
