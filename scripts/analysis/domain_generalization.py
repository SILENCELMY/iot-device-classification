#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Generalization for IoT Device Identification
Focus: Robust identification under distribution shift

Methods:
1. CORAL (CORrelation ALignment) - feature covariance shift mitigation
2. Baseline RF - no adaptation
3. Domain-aware feature selection

Priority: R5, R6, R7, LORO cross-scenario generalization
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler


class CORAL:
    """
    CORAL: CORrelation ALignment

    Sun et al., "Return of Frustratingly Easy Domain Adaptation", AAAI 2016

    Aligns second-order statistics (covariance) between source and target domains.
    Aims to mitigate feature covariance shift.
    """

    def __init__(self, lambda_coral: float = 1.0):
        """
        Args:
            lambda_coral: Weight for CORAL loss (default 1.0 for full alignment)
        """
        self.lambda_coral = lambda_coral
        self.source_cov = None
        self.target_cov = None
        self.transform_matrix = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORAL':
        """
        Compute CORAL transformation matrix.

        Args:
            X_source: Source domain features (n_source, n_features)
            X_target: Target domain features (n_target, n_features)

        Returns:
            self
        """
        # Store means for later use
        self.source_mean = X_source.mean(axis=0)
        self.target_mean = X_target.mean(axis=0)

        # Center the features
        X_source_centered = X_source - self.source_mean
        X_target_centered = X_target - self.target_mean

        # Compute covariance matrices with regularization
        # Add stronger regularization for numerical stability
        n_features = X_source.shape[1]
        reg = 1e-3  # Stronger regularization

        self.source_cov = np.cov(X_source_centered, rowvar=False) + np.eye(n_features) * reg
        self.target_cov = np.cov(X_target_centered, rowvar=False) + np.eye(n_features) * reg

        # Compute whitening and coloring transformation
        # Transform: X' = X * C_s^(-1/2) * C_t^(1/2)

        # Source whitening: C_s^(-1/2)
        eigval_s, eigvec_s = np.linalg.eigh(self.source_cov)
        eigval_s = np.maximum(eigval_s, 1e-3)  # Stronger clipping
        source_whitening = eigvec_s @ np.diag(eigval_s ** (-0.5)) @ eigvec_s.T

        # Target coloring: C_t^(1/2)
        eigval_t, eigvec_t = np.linalg.eigh(self.target_cov)
        eigval_t = np.maximum(eigval_t, 1e-3)
        target_coloring = eigvec_t @ np.diag(eigval_t ** 0.5) @ eigvec_t.T

        # Combined transformation
        self.transform_matrix = source_whitening @ target_coloring

        return self

    def transform(self, X: np.ndarray, use_source_mean: bool = False) -> np.ndarray:
        """
        Apply CORAL transformation.

        Args:
            X: Features to transform (n_samples, n_features)
            use_source_mean: If True, center using source mean (for source domain)
                            If False, center using own mean (for target domain)

        Returns:
            Transformed features
        """
        if self.transform_matrix is None:
            raise ValueError("CORAL not fitted. Call fit() first.")

        # Center using appropriate mean
        if use_source_mean:
            X_centered = X - self.source_mean
        else:
            X_centered = X - X.mean(axis=0)

        X_transformed = X_centered @ self.transform_matrix

        # Weighted alignment
        return self.lambda_coral * X_transformed + (1 - self.lambda_coral) * X_centered

    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Fit CORAL and transform source domain.

        Args:
            X_source: Source domain features
            X_target: Target domain features (for computing statistics only)

        Returns:
            Transformed source features
        """
        self.fit(X_source, X_target)
        return self.transform(X_source, use_source_mean=True)


def compute_coral_loss(X_source: np.ndarray, X_target: np.ndarray) -> float:
    """
    Compute CORAL loss (Frobenius norm of covariance difference).

    Lower is better (closer covariance structures).

    Args:
        X_source: Source features
        X_target: Target features

    Returns:
        CORAL loss value
    """
    X_s = X_source - X_source.mean(axis=0)
    X_t = X_target - X_target.mean(axis=0)

    cov_s = np.cov(X_s, rowvar=False)
    cov_t = np.cov(X_t, rowvar=False)

    loss = np.linalg.norm(cov_s - cov_t, 'fro') ** 2
    loss /= (4 * X_source.shape[1] ** 2)  # Normalize

    return loss


def train_baseline_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train baseline Random Forest without domain adaptation.

    Returns:
        Dictionary with model, predictions, and metrics
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    return {
        'model': clf,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def train_coral_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_coral: float = 1.0,
    n_estimators: int = 200,
    random_state: int = 42,
    check_alignment: bool = True
) -> Dict[str, Any]:
    """
    Train Random Forest with CORAL domain adaptation.

    Returns:
        Dictionary with CORAL transformer, model, predictions, and metrics
    """
    # Compute CORAL loss before alignment
    coral_loss_before = compute_coral_loss(X_train, X_test)

    # Apply CORAL alignment
    coral = CORAL(lambda_coral=lambda_coral)
    X_train_aligned = coral.fit_transform(X_train, X_test)  # source domain
    X_test_aligned = coral.transform(X_test, use_source_mean=False)  # target domain

    # Compute CORAL loss after alignment
    coral_loss_after = compute_coral_loss(X_train_aligned, X_test_aligned)

    # Optional: Check if alignment makes sense
    if check_alignment and coral_loss_after > coral_loss_before * 2:
        print(f"  [WARNING] CORAL increased loss: {coral_loss_before:.6f} → {coral_loss_after:.6f}")
        print(f"  This suggests feature scale issues. Using λ=0.5 instead...")

        # Retry with reduced alignment strength
        coral = CORAL(lambda_coral=0.5)
        X_train_aligned = coral.fit_transform(X_train, X_test)
        X_test_aligned = coral.transform(X_test, use_source_mean=False)
        coral_loss_after = compute_coral_loss(X_train_aligned, X_test_aligned)

    # Train RF on aligned features
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    clf.fit(X_train_aligned, y_train)
    y_pred = clf.predict(X_test_aligned)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    return {
        'coral': coral,
        'model': clf,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'coral_loss_before': coral_loss_before,
        'coral_loss_after': coral_loss_after,
        'coral_loss_reduction': coral_loss_before - coral_loss_after
    }


def load_features(
    results_root: Path,
    filter_mode: str = 'raw_all'
) -> pd.DataFrame:
    """
    Load pre-extracted features from robust_v2 results.

    Args:
        results_root: Path to results directory (e.g., results/robust_v2)
        filter_mode: Filter mode (default 'raw_all')

    Returns:
        DataFrame with features and metadata
    """
    feature_cache = results_root / filter_mode / f"features_{filter_mode}_w10.csv"

    if not feature_cache.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_cache}")

    print(f"[INFO] Loading features from {feature_cache}")
    df = pd.read_csv(feature_cache)
    print(f"[INFO] Loaded {len(df)} samples with {len(df.columns)} columns")

    return df


def prepare_task_data(
    df: pd.DataFrame,
    train_rounds: List[str],
    test_rounds: List[str],
    meta_columns: set = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare train/test split for a cross-scenario task.

    Args:
        df: Feature DataFrame
        train_rounds: Training rounds (e.g., ['R2', 'R3', 'R4'])
        test_rounds: Testing rounds (e.g., ['R5'])
        meta_columns: Columns to exclude from features

    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    if meta_columns is None:
        meta_columns = {
            'label', 'round', 'traffic', 'filter_mode',
            'source_file', 'window_id', 'window_start', 'window_end'
        }

    # Split by rounds
    train_mask = df['round'].isin(train_rounds)
    test_mask = df['round'].isin(test_rounds)

    df_train = df[train_mask]
    df_test = df[test_mask]

    print(f"[INFO] Train: {len(df_train)} samples from {train_rounds}")
    print(f"[INFO] Test: {len(df_test)} samples from {test_rounds}")

    # Extract features and labels
    feature_cols = [col for col in df.columns if col not in meta_columns]

    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values

    # Standardize (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, feature_cols


def run_dg_experiment(
    results_root: Path,
    output_root: Path,
    task_name: str,
    train_rounds: List[str],
    test_rounds: List[str],
    lambda_coral: float = 1.0,
    n_estimators: int = 200,
    random_state: int = 42
):
    """
    Run domain generalization experiment: Baseline RF vs CORAL+RF.

    Args:
        results_root: Path to robust_v2 results (for feature loading)
        output_root: Path to save DG results
        task_name: Task identifier (e.g., 'position_R2R3R4_to_R5')
        train_rounds: Training rounds
        test_rounds: Testing rounds
        lambda_coral: CORAL alignment strength
        n_estimators: Number of RF trees
        random_state: Random seed
    """
    print("\n" + "=" * 80)
    print(f"Domain Generalization Experiment: {task_name}".center(80))
    print("=" * 80)

    # Load features
    df = load_features(results_root)

    # Prepare data
    X_train, y_train, X_test, y_test, feature_names = prepare_task_data(
        df, train_rounds, test_rounds
    )

    # Get class names
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))

    # Create output directory
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # ====================================================================
    # Baseline: RF without adaptation
    # ====================================================================
    print("\n[1/2] Training Baseline RF (no adaptation)...")
    baseline_result = train_baseline_rf(
        X_train, y_train, X_test, y_test,
        n_estimators=n_estimators,
        random_state=random_state
    )

    print(f"  Accuracy: {baseline_result['accuracy']:.4f}")
    print(f"  Macro-F1: {baseline_result['macro_f1']:.4f}")

    # Save baseline results
    baseline_dir = task_dir / "baseline_rf"
    baseline_dir.mkdir(exist_ok=True)

    joblib.dump(baseline_result['model'], baseline_dir / "model.joblib")
    pd.DataFrame(
        baseline_result['confusion_matrix'],
        index=classes,
        columns=classes
    ).to_csv(baseline_dir / "confusion_matrix.csv")

    with open(baseline_dir / "metrics.json", 'w') as f:
        json.dump({
            'task': task_name,
            'method': 'baseline_rf',
            'train_rounds': train_rounds,
            'test_rounds': test_rounds,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_estimators': n_estimators,
            'accuracy': float(baseline_result['accuracy']),
            'precision': float(baseline_result['precision']),
            'recall': float(baseline_result['recall']),
            'macro_f1': float(baseline_result['macro_f1'])
        }, f, indent=2)

    # ====================================================================
    # CORAL + RF
    # ====================================================================
    print(f"\n[2/2] Training CORAL + RF (λ={lambda_coral})...")
    coral_result = train_coral_rf(
        X_train, y_train, X_test, y_test,
        lambda_coral=lambda_coral,
        n_estimators=n_estimators,
        random_state=random_state
    )

    print(f"  Accuracy: {coral_result['accuracy']:.4f}")
    print(f"  Macro-F1: {coral_result['macro_f1']:.4f}")
    print(f"  CORAL Loss: {coral_result['coral_loss_before']:.6f} → {coral_result['coral_loss_after']:.6f}")
    print(f"  Loss Reduction: {coral_result['coral_loss_reduction']:.6f}")

    # Save CORAL results
    coral_dir = task_dir / "coral_rf"
    coral_dir.mkdir(exist_ok=True)

    joblib.dump(coral_result['coral'], coral_dir / "coral_transformer.joblib")
    joblib.dump(coral_result['model'], coral_dir / "model.joblib")
    pd.DataFrame(
        coral_result['confusion_matrix'],
        index=classes,
        columns=classes
    ).to_csv(coral_dir / "confusion_matrix.csv")

    with open(coral_dir / "metrics.json", 'w') as f:
        json.dump({
            'task': task_name,
            'method': 'coral_rf',
            'train_rounds': train_rounds,
            'test_rounds': test_rounds,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'lambda_coral': lambda_coral,
            'n_estimators': n_estimators,
            'accuracy': float(coral_result['accuracy']),
            'precision': float(coral_result['precision']),
            'recall': float(coral_result['recall']),
            'macro_f1': float(coral_result['macro_f1']),
            'coral_loss_before': float(coral_result['coral_loss_before']),
            'coral_loss_after': float(coral_result['coral_loss_after']),
            'coral_loss_reduction': float(coral_result['coral_loss_reduction'])
        }, f, indent=2)

    # ====================================================================
    # Comparison Report
    # ====================================================================
    print("\n" + "=" * 80)
    print("Comparison Summary".center(80))
    print("=" * 80)

    improvement = coral_result['macro_f1'] - baseline_result['macro_f1']
    print(f"\nBaseline RF:  {baseline_result['macro_f1']:.4f}")
    print(f"CORAL + RF:   {coral_result['macro_f1']:.4f}")
    print(f"Improvement:  {improvement:+.4f} ({improvement/baseline_result['macro_f1']*100:+.2f}%)")

    # Save comparison
    comparison = {
        'task': task_name,
        'train_rounds': train_rounds,
        'test_rounds': test_rounds,
        'baseline_rf_macro_f1': float(baseline_result['macro_f1']),
        'coral_rf_macro_f1': float(coral_result['macro_f1']),
        'improvement_absolute': float(improvement),
        'improvement_relative_pct': float(improvement / baseline_result['macro_f1'] * 100),
        'coral_loss_reduction': float(coral_result['coral_loss_reduction'])
    }

    with open(task_dir / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[INFO] Results saved to {task_dir}/")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Domain Generalization: CORAL + RF for robust IoT identification"
    )
    parser.add_argument(
        "--results-root",
        default="../results/robust_v2",
        help="Path to robust_v2 results (for feature loading)"
    )
    parser.add_argument(
        "--output-root",
        default="../results/domain_generalization",
        help="Output directory for DG results"
    )
    parser.add_argument(
        "--tasks",
        default="position,jitter,loro",
        help="Comma-separated tasks: position, jitter_r6, jitter_r7, loro_all, loro_r3"
    )
    parser.add_argument(
        "--lambda-coral",
        type=float,
        default=1.0,
        help="CORAL alignment strength (0=no alignment, 1=full alignment)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in Random Forest"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)

    # Define priority tasks
    TASKS = {
        'position': {
            'name': 'position_R2R3R4_to_R5',
            'train': ['R2', 'R3', 'R4'],
            'test': ['R5']
        },
        'jitter_r6': {
            'name': 'jitter_R2R3R4_to_R6',
            'train': ['R2', 'R3', 'R4'],
            'test': ['R6']
        },
        'jitter_r7': {
            'name': 'jitter_R2R3R4_to_R7',
            'train': ['R2', 'R3', 'R4'],
            'test': ['R7']
        },
        'loro_r3': {
            'name': 'loro_R2R4_to_R3',
            'train': ['R2', 'R4'],
            'test': ['R3']
        },
        'loro_r4': {
            'name': 'loro_R2R3_to_R4',
            'train': ['R2', 'R3'],
            'test': ['R4']
        },
        'loro_r2': {
            'name': 'loro_R3R4_to_R2',
            'train': ['R3', 'R4'],
            'test': ['R2']
        }
    }

    requested_tasks = [t.strip() for t in args.tasks.split(',')]

    # Expand shortcuts
    if 'position' in requested_tasks:
        requested_tasks.remove('position')
        requested_tasks.append('position')
    if 'jitter' in requested_tasks:
        requested_tasks.remove('jitter')
        requested_tasks.extend(['jitter_r6', 'jitter_r7'])
    if 'loro' in requested_tasks:
        requested_tasks.remove('loro')
        requested_tasks.extend(['loro_r2', 'loro_r3', 'loro_r4'])
    if 'loro_all' in requested_tasks:
        requested_tasks.remove('loro_all')
        requested_tasks.extend(['loro_r2', 'loro_r3', 'loro_r4'])

    # Run experiments
    for task_key in requested_tasks:
        if task_key not in TASKS:
            print(f"[WARNING] Unknown task: {task_key}")
            continue

        task_config = TASKS[task_key]
        run_dg_experiment(
            results_root=results_root,
            output_root=output_root,
            task_name=task_config['name'],
            train_rounds=task_config['train'],
            test_rounds=task_config['test'],
            lambda_coral=args.lambda_coral,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )


if __name__ == '__main__':
    main()
