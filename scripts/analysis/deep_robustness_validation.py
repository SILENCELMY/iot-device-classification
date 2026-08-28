#!/usr/bin/env python3
"""Small deep-learning robustness validation for the IoT feature pipeline.

This script intentionally keeps the experiment small. It reuses the existing
10-second window feature cache and task definitions, trains lightweight neural
models on the same splits, and compares them with existing RF/XGBoost/Stacking
results from the main robust_v2 run.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT_DEPS = True
except ModuleNotFoundError:
    plt = None
    sns = None
    HAS_PLOT_DEPS = False


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

from robust_iot_research import (  # noqa: E402
    clean_x,
    feature_columns,
    fit_label_encoder,
    metric_summary,
    read_config,
    task_data,
)


TASKS = [
    "single_round_R2",
    "single_round_R3",
    "single_round_R4",
    "loro_R2_R3_to_R4",
    "loro_R2_R4_to_R3",
    "loro_R3_R4_to_R2",
    "position_R2_R3_R4_to_R5",
    "jitter_R2_R3_R4_to_R6_R7",
]

TRADITIONAL_MODELS = ["rf"]
DEEP_MODELS = [
    "cnn1d",
    "transformer",
    "cnn1d_v2",
    "cnn1d_v25",
    "cnn1d_v3",
    "cnn1d_v3_sharp",
    "cnn1d_v3_smooth",
    "cnn1d_v3_hybrid",
    "cnn1d_inception",
    "cnn1d_tcn",
    "cnn1d_convnext",
    "cnn1d_v4",
    "cnn1d_v5",
    "transformer_v2",
]
CPD_MODELS = [
    "rf",
    "cnn1d",
    "cnn1d_v2",
    "cnn1d_v25",
    "cnn1d_v3",
    "cnn1d_v3_sharp",
    "cnn1d_v3_smooth",
    "cnn1d_v3_hybrid",
    "cnn1d_inception",
    "cnn1d_tcn",
    "cnn1d_convnext",
    "cnn1d_v4",
    "cnn1d_v5",
    "transformer",
    "transformer_v2",
]
CONFUSION_MODELS = [
    "rf",
    "cnn1d",
    "cnn1d_v2",
    "cnn1d_v25",
    "cnn1d_v3",
    "cnn1d_v3_sharp",
    "cnn1d_v3_smooth",
    "cnn1d_v3_hybrid",
    "cnn1d_inception",
    "cnn1d_tcn",
    "cnn1d_convnext",
    "cnn1d_v4",
    "cnn1d_v5",
    "transformer",
    "transformer_v2",
]
LABELS = ["Camera", "Light_T1", "Light_XM", "Sensor", "Socket"]
KEY_CLASSES = ["Sensor", "Light_T1", "Light_XM"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabularCNN1D(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(64, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.10) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class SEAttention1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(x).unsqueeze(-1)
        return x * weights


class TabularCNN1Dv2(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(128, kernel_size=3),
            ResidualConvBlock(128, kernel_size=3),
        )
        self.se = SEAttention1D(128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.se(x)
        return self.head(x)


class TabularCNN1Dv25(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 144, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(144, branch_channels=40),
            MultiScaleResidualConvBlock(144, branch_channels=40),
        )
        self.attention_pool = AttentionPool1D(144)
        self.head = nn.Sequential(
            nn.LayerNorm(144 * 2),
            nn.Dropout(0.22),
            nn.Linear(144 * 2, 160),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).unsqueeze(1)
        x = self.blocks(self.stem(x))
        pooled = torch.cat([self.attention_pool(x), x.mean(dim=-1)], dim=1)
        return self.head(pooled)


class MultiScaleResidualConvBlock(nn.Module):
    def __init__(self, channels: int, branch_channels: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1),
                nn.Conv1d(channels, branch_channels, kernel_size=5, padding=2),
                nn.Conv1d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            ]
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(branch_channels * 3, channels, kernel_size=1),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.se = SEAttention1D(channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.se(self.mix(multi_scale)))


class AttentionPool1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=-1)
        return (x * weights).sum(dim=-1)


class FeatureSelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4, dropout: float = 0.08) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.attn_scale = nn.Parameter(torch.tensor(0.10))
        self.ffn_scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        attn_in = self.norm_attn(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + self.attn_scale * attn_out
        tokens = tokens + self.ffn_scale * self.ffn(self.norm_ffn(tokens))
        return tokens.transpose(1, 2)


class TabularCNN1Dv3(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 160, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(160, branch_channels=64),
            MultiScaleResidualConvBlock(160, branch_channels=64),
            MultiScaleResidualConvBlock(160, branch_channels=64),
        )
        self.attention_pool = AttentionPool1D(160)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 160),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(160 * 4),
            nn.Dropout(0.25),
            nn.Linear(160 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.blocks(self.stem(x))
        pooled = torch.cat(
            [
                self.attention_pool(x),
                x.mean(dim=-1),
                x.amax(dim=-1),
                raw,
            ],
            dim=1,
        )
        return self.head(pooled)


class TabularCNN1Dv3Sharp(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 72, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(72, 144, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(144, 176, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(176, branch_channels=72, dropout=0.03),
            MultiScaleResidualConvBlock(176, branch_channels=72, dropout=0.03),
            MultiScaleResidualConvBlock(176, branch_channels=72, dropout=0.03),
            FeatureSelfAttentionBlock(176, heads=4, dropout=0.03),
        )
        self.se = SEAttention1D(176, reduction=6)
        self.attention_pool = AttentionPool1D(176)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 176),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(176 * 5),
            nn.Dropout(0.08),
            nn.Linear(176 * 5, 384),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Linear(192, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.se(self.blocks(self.stem(x)))
        pooled = torch.cat(
            [
                self.attention_pool(x),
                x.mean(dim=-1),
                x.amax(dim=-1),
                x.std(dim=-1, unbiased=False),
                raw,
            ],
            dim=1,
        )
        return self.head(pooled)


class TabularCNN1Dv3Smooth(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 56, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(56, 112, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(112, 152, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(152, branch_channels=56, dropout=0.18),
            MultiScaleResidualConvBlock(152, branch_channels=56, dropout=0.18),
            MultiScaleResidualConvBlock(152, branch_channels=56, dropout=0.18),
        )
        self.se = SEAttention1D(152)
        self.attention_pool = AttentionPool1D(152)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Dropout(0.12),
            nn.Linear(feature_count, 152),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(152 * 3),
            nn.Dropout(0.34),
            nn.Linear(152 * 3, 192),
            nn.GELU(),
            nn.Dropout(0.24),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(96, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.se(self.blocks(self.stem(x)))
        pooled = torch.cat([self.attention_pool(x), x.mean(dim=-1), raw], dim=1)
        return self.head(pooled)


class TabularCNN1Dv3Hybrid(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 168, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(168, branch_channels=64, dropout=0.10),
            MultiScaleResidualConvBlock(168, branch_channels=64, dropout=0.10),
            MultiScaleResidualConvBlock(168, branch_channels=64, dropout=0.10),
            FeatureSelfAttentionBlock(168, heads=4, dropout=0.08),
        )
        self.se = SEAttention1D(168)
        self.attention_pool = AttentionPool1D(168)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 168),
            nn.GELU(),
            nn.Dropout(0.06),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(168 * 4),
            nn.Dropout(0.18),
            nn.Linear(168 * 4, 288),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(288, 144),
            nn.GELU(),
            nn.Linear(144, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.se(self.blocks(self.stem(x)))
        pooled = torch.cat([self.attention_pool(x), x.mean(dim=-1), x.amax(dim=-1), raw], dim=1)
        return self.head(pooled)


class InceptionResidualBlock1D(nn.Module):
    def __init__(self, channels: int, branch_channels: int = 48, dropout: float = 0.12) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(channels, branch_channels, kernel_size=1), nn.GELU()),
                nn.Sequential(nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1), nn.GELU()),
                nn.Sequential(nn.Conv1d(channels, branch_channels, kernel_size=5, padding=2), nn.GELU()),
                nn.Sequential(
                    nn.Conv1d(channels, branch_channels, kernel_size=3, padding=3, dilation=3),
                    nn.GELU(),
                ),
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(branch_channels * 4, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.se = SEAttention1D(channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.activation(x + self.se(self.mix(mixed)))


class TabularCNNInception(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 144, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            InceptionResidualBlock1D(144, branch_channels=48, dropout=0.14),
            InceptionResidualBlock1D(144, branch_channels=48, dropout=0.14),
            InceptionResidualBlock1D(144, branch_channels=48, dropout=0.14),
        )
        self.attention_pool = AttentionPool1D(144)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Dropout(0.10),
            nn.Linear(feature_count, 144),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(144 * 4),
            nn.Dropout(0.26),
            nn.Linear(144 * 4, 224),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(224, 112),
            nn.GELU(),
            nn.Linear(112, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.blocks(self.stem(self.norm(x).unsqueeze(1)))
        pooled = torch.cat(
            [self.attention_pool(x), x.mean(dim=-1), x.amax(dim=-1), raw],
            dim=1,
        )
        return self.head(pooled)


class TCNResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.12) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
        )
        self.se = SEAttention1D(channels, reduction=10)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.se(self.net(x)))


class TabularCNNTCN(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(80, 144, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            TCNResidualBlock1D(144, dilation=1, dropout=0.14),
            TCNResidualBlock1D(144, dilation=2, dropout=0.14),
            TCNResidualBlock1D(144, dilation=4, dropout=0.14),
            TCNResidualBlock1D(144, dilation=8, dropout=0.14),
        )
        self.attention_pool = AttentionPool1D(144)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 144),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(144 * 4),
            nn.Dropout(0.28),
            nn.Linear(144 * 4, 224),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(224, 112),
            nn.GELU(),
            nn.Linear(112, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.blocks(self.stem(self.norm(x).unsqueeze(1)))
        pooled = torch.cat(
            [self.attention_pool(x), x.mean(dim=-1), x.std(dim=-1, unbiased=False), raw],
            dim=1,
        )
        return self.head(pooled)


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pointwise = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * expansion, channels),
        )
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-3)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.pointwise(self.norm(x)) * self.gamma
        x = x.transpose(1, 2)
        return self.activation(residual + x)


class TabularCNNConvNeXt(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 160, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ConvNeXtBlock1D(160, expansion=4, dropout=0.12),
            ConvNeXtBlock1D(160, expansion=4, dropout=0.12),
            ConvNeXtBlock1D(160, expansion=4, dropout=0.12),
            ConvNeXtBlock1D(160, expansion=4, dropout=0.12),
        )
        self.se = SEAttention1D(160, reduction=8)
        self.attention_pool = AttentionPool1D(160)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Dropout(0.08),
            nn.Linear(feature_count, 160),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(160 * 4),
            nn.Dropout(0.24),
            nn.Linear(160 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.se(self.blocks(self.stem(self.norm(x).unsqueeze(1))))
        pooled = torch.cat(
            [self.attention_pool(x), x.mean(dim=-1), x.amax(dim=-1), raw],
            dim=1,
        )
        return self.head(pooled)


class TabularCNN1Dv4(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 80, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(80, 160, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(160, 192, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(192, branch_channels=72, dropout=0.06),
            MultiScaleResidualConvBlock(192, branch_channels=72, dropout=0.06),
            MultiScaleResidualConvBlock(192, branch_channels=72, dropout=0.06),
            FeatureSelfAttentionBlock(192, heads=4, dropout=0.06),
            MultiScaleResidualConvBlock(192, branch_channels=72, dropout=0.06),
        )
        self.se = SEAttention1D(192, reduction=6)
        self.attention_pool = AttentionPool1D(192)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 192),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(192 * 5),
            nn.Dropout(0.18),
            nn.Linear(192 * 5, 384),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(0.06),
            nn.Linear(192, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.se(self.blocks(self.stem(x)))
        pooled = torch.cat(
            [
                self.attention_pool(x),
                x.mean(dim=-1),
                x.amax(dim=-1),
                x.std(dim=-1, unbiased=False),
                raw,
            ],
            dim=1,
        )
        return self.head(pooled)


class TabularCNN1Dv5(nn.Module):
    def __init__(self, feature_count: int, class_count: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_count)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 160, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            MultiScaleResidualConvBlock(160, branch_channels=64, dropout=0.08),
            MultiScaleResidualConvBlock(160, branch_channels=64, dropout=0.08),
            MultiScaleResidualConvBlock(160, branch_channels=64, dropout=0.08),
            FeatureSelfAttentionBlock(160, heads=4, dropout=0.08),
        )
        self.se = SEAttention1D(160)
        self.attention_pool = AttentionPool1D(160)
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, 160),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(160 * 4),
            nn.Dropout(0.22),
            nn.Linear(160 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.14),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_skip(x)
        x = self.norm(x).unsqueeze(1)
        x = self.se(self.blocks(self.stem(x)))
        pooled = torch.cat(
            [
                self.attention_pool(x),
                x.mean(dim=-1),
                x.amax(dim=-1),
                raw,
            ],
            dim=1,
        )
        return self.head(pooled)


class LightweightTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
        class_count: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, feature_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.15),
            nn.Linear(d_model, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(x.unsqueeze(-1)) + self.position
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class LightweightTransformerv2(nn.Module):
    def __init__(
        self,
        feature_count: int,
        class_count: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, feature_count, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.raw_skip = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Dropout(0.20),
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_embedding(x.unsqueeze(-1)) + self.position
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.attn_pool(encoded), dim=1)
        pooled = torch.cat(
            [(encoded * weights).sum(dim=1), encoded.mean(dim=1), self.raw_skip(x)],
            dim=1,
        )
        return self.head(pooled)


def build_deep_model(name: str, feature_count: int, class_count: int) -> nn.Module:
    if name == "cnn1d":
        return TabularCNN1D(feature_count, class_count)
    if name == "cnn1d_v2":
        return TabularCNN1Dv2(feature_count, class_count)
    if name == "cnn1d_v25":
        return TabularCNN1Dv25(feature_count, class_count)
    if name == "cnn1d_v3":
        return TabularCNN1Dv3(feature_count, class_count)
    if name == "cnn1d_v3_sharp":
        return TabularCNN1Dv3Sharp(feature_count, class_count)
    if name == "cnn1d_v3_smooth":
        return TabularCNN1Dv3Smooth(feature_count, class_count)
    if name == "cnn1d_v3_hybrid":
        return TabularCNN1Dv3Hybrid(feature_count, class_count)
    if name == "cnn1d_inception":
        return TabularCNNInception(feature_count, class_count)
    if name == "cnn1d_tcn":
        return TabularCNNTCN(feature_count, class_count)
    if name == "cnn1d_convnext":
        return TabularCNNConvNeXt(feature_count, class_count)
    if name == "cnn1d_v4":
        return TabularCNN1Dv4(feature_count, class_count)
    if name == "cnn1d_v5":
        return TabularCNN1Dv5(feature_count, class_count)
    if name == "transformer":
        return LightweightTransformer(feature_count, class_count)
    if name == "transformer_v2":
        return LightweightTransformerv2(feature_count, class_count)
    raise ValueError(f"Unsupported deep model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_deep_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    args: argparse.Namespace,
    class_count: int,
) -> tuple[np.ndarray, nn.Module, StandardScaler]:
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train).astype(np.float32)
    x_test_np = scaler.transform(x_test).astype(np.float32)
    y_train_np = y_train.astype(np.int64)

    model = build_deep_model(model_name, x_train_np.shape[1], class_count).to(args.device)
    class_counts = np.bincount(y_train_np, minlength=class_count).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=args.device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(
        torch.from_numpy(x_train_np),
        torch.from_numpy(y_train_np),
    )
    generator = torch.Generator()
    generator.manual_seed(args.random_state)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    model.eval()
    preds: list[np.ndarray] = []
    test_tensor = torch.from_numpy(x_test_np)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in test_loader:
            logits = model(xb.to(args.device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), model, scaler


def normalized_offdiag(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm = cm / row_sums
    np.fill_diagonal(norm, 0.0)
    return norm


def compute_cpd(cm1: np.ndarray, cm2: np.ndarray) -> float:
    return float(np.linalg.norm(normalized_offdiag(cm1) - normalized_offdiag(cm2), ord="fro"))


def task_scenario(task: str) -> str:
    if task.startswith("single_round"):
        return "IID"
    if task.startswith("loro"):
        return "OOD_LORO"
    if task.startswith("position"):
        return "OOD_Position"
    if task.startswith("jitter"):
        return "OOD_Jitter"
    return "Other"


def prepare_task_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["name"]: task for task in config["evaluation_tasks"]}


def parse_model_list(value: str, default: list[str]) -> list[str]:
    if value.strip().lower() in {"", "none"}:
        return []
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or default


def load_existing_result(
    source_root: Path,
    model_name: str,
    task_name: str,
    output_dir: Path,
    source_label: str,
) -> dict[str, Any] | None:
    src_dir = source_root / task_name / "all_features" / model_name
    metrics_path = src_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Skipping missing existing result: {src_dir}", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "metrics.json",
        "classification_report.csv",
        "confusion_matrix.csv",
        "predictions.csv",
        "feature_columns.json",
        "model.pt",
        "scaler.joblib",
    ]:
        src = src_dir / filename
        if src.exists():
            dst = output_dir / filename
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = dict(metrics)
    metrics["model"] = model_name
    metrics["source"] = source_label
    return metrics


def load_traditional_result(
    model_name: str,
    task_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    return load_existing_result(
        source_root=args.traditional_source,
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        source_label="traditional_source",
    )


def load_deep_source_result(
    model_name: str,
    task_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    if args.deep_source is None:
        return None
    return load_existing_result(
        source_root=args.deep_source / "raw_all",
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        source_label="deep_source",
    )


def evaluate_deep_task(
    model_name: str,
    features: pd.DataFrame,
    task: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    split_args = SimpleNamespace(
        test_size=args.test_size,
        random_state=args.random_state,
        max_rows=args.max_rows,
    )
    train_data, test_data, y_train, y_test, _, meta_test = task_data(features, task, split_args)
    columns = feature_columns(train_data)
    x_train = clean_x(train_data, columns)
    x_test = clean_x(test_data, columns)
    encoder = fit_label_encoder(config["labels"])
    y_train_encoded = encoder.transform(y_train)

    pred_encoded, model, scaler = train_deep_model(
        model_name=model_name,
        x_train=x_train,
        y_train=y_train_encoded,
        x_test=x_test,
        args=args,
        class_count=len(config["labels"]),
    )
    pred_labels = encoder.inverse_transform(pred_encoded.astype(int))

    metrics, cm, report = metric_summary(y_test.to_numpy(), pred_labels, config["labels"])
    predictions = meta_test.copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = pred_labels
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", encoding="utf-8-sig")
    pd.DataFrame(cm, index=config["labels"], columns=config["labels"]).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2, ensure_ascii=False)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    torch.save(model.state_dict(), output_dir / "model.pt")

    summary = {
        "filter_mode": "raw_all",
        "task": task["name"],
        "task_type": task["type"],
        "scenario": task_scenario(task["name"]),
        "train_rounds": task.get("train_rounds", task.get("rounds", [])),
        "test_rounds": task.get("test_rounds", task.get("rounds", [])),
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
        "feature_set": "all_features",
        "model": model_name,
        "feature_count": len(columns),
        "parameter_count": parameter_count(model),
        **{key: value for key, value in metrics.items() if not key.startswith("per_class")},
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": cm.tolist(),
        "source": "trained_deep_validation",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def load_confusion_matrix(result_root: Path, task: str, model: str) -> np.ndarray | None:
    path = result_root / "raw_all" / task / "all_features" / model / "confusion_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0).values


def build_performance_tables(summaries: pd.DataFrame, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    iid = (
        summaries[summaries["scenario"] == "IID"]
        .groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "iid_mean_macro_f1"})
    )
    ood = (
        summaries[summaries["scenario"] != "IID"]
        .groupby("model", as_index=False)["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "ood_mean_macro_f1"})
    )
    perf = summaries.merge(iid, on="model", how="left")
    perf["ood_drop"] = np.where(
        perf["scenario"] == "IID",
        0.0,
        perf["iid_mean_macro_f1"] - perf["macro_f1"],
    )
    perf.to_csv(report_dir / "performance_comparison.csv", index=False, encoding="utf-8-sig")

    model_summary = iid.merge(ood, on="model", how="left")
    model_summary["mean_ood_drop"] = model_summary["iid_mean_macro_f1"] - model_summary["ood_mean_macro_f1"]
    model_summary.to_csv(report_dir / "iid_ood_model_summary.csv", index=False, encoding="utf-8-sig")
    return perf, model_summary


def build_cpd_table(result_root: Path, report_dir: Path, tasks: list[str]) -> pd.DataFrame:
    iid_tasks = ["single_round_R2", "single_round_R3", "single_round_R4"]
    ood_tasks = [task for task in tasks if task not in iid_tasks]
    rows = []
    for model in CPD_MODELS:
        iid_cms = [load_confusion_matrix(result_root, task, model) for task in iid_tasks]
        iid_cms = [cm for cm in iid_cms if cm is not None]
        for task in ood_tasks:
            ood_cm = load_confusion_matrix(result_root, task, model)
            if ood_cm is None or not iid_cms:
                continue
            cpds = [compute_cpd(iid_cm, ood_cm) for iid_cm in iid_cms]
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "scenario": task_scenario(task),
                    "cpd_vs_iid_mean": float(np.mean(cpds)),
                    "cpd_vs_iid_max": float(np.max(cpds)),
                }
            )
    cpd = pd.DataFrame(rows, columns=["model", "task", "scenario", "cpd_vs_iid_mean", "cpd_vs_iid_max"])
    cpd.to_csv(report_dir / "cpd_comparison.csv", index=False, encoding="utf-8-sig")
    return cpd


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无可用结果_"
    shown = df.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.to_numpy()]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def plot_degradation(perf: pd.DataFrame, report_dir: Path) -> None:
    if not HAS_PLOT_DEPS:
        print("Skipping plots: matplotlib/seaborn are not available in this Python environment.", flush=True)
        return
    data = perf[perf["scenario"] != "IID"].copy()
    data["task_short"] = data["task"].str.replace("jitter_R2_R3_R4_to_", "jitter_", regex=False)
    data["task_short"] = data["task_short"].str.replace("position_R2_R3_R4_to_", "position_", regex=False)
    data["task_short"] = data["task_short"].str.replace("loro_", "loro_", regex=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="task_short", y="ood_drop", hue="model")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("OOD drop = IID mean macro-F1 - task macro-F1")
    plt.xlabel("OOD task")
    plt.title("IID/OOD performance degradation")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "iid_ood_degradation.png", dpi=220)
    plt.close()


def plot_cpd(cpd: pd.DataFrame, report_dir: Path) -> None:
    if not HAS_PLOT_DEPS:
        return
    plt.figure(figsize=(9, 5))
    sns.barplot(data=cpd, x="model", y="cpd_vs_iid_mean", errorbar="sd")
    plt.ylabel("Mean CPD vs IID")
    plt.xlabel("Model")
    plt.title("Confusion topology drift comparison")
    plt.tight_layout()
    plt.savefig(report_dir / "cpd_comparison.png", dpi=220)
    plt.close()


def plot_confusion_topology(result_root: Path, report_dir: Path) -> None:
    if not HAS_PLOT_DEPS:
        return
    tasks = ["single_round_R3", "loro_R2_R4_to_R3", "position_R2_R3_R4_to_R5", "jitter_R2_R3_R4_to_R6_R7"]
    models = CONFUSION_MODELS
    fig, axes = plt.subplots(len(models), len(tasks), figsize=(16, 14))
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            cm = load_confusion_matrix(result_root, task, model)
            if cm is None:
                ax.axis("off")
                continue
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums
            sns.heatmap(
                cm_norm,
                ax=ax,
                cmap="Blues",
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt=".2f",
                annot_kws={"fontsize": 7},
            )
            ax.set_title(f"{model} / {task}", fontsize=9)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_topology_comparison.png", dpi=220)
    plt.close()


def build_robustness_ranking(model_summary: pd.DataFrame, cpd: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    if cpd.empty:
        cpd_summary = pd.DataFrame(columns=["model", "mean_cpd_vs_iid"])
    else:
        cpd_summary = (
            cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"]
            .mean()
            .rename(columns={"cpd_vs_iid_mean": "mean_cpd_vs_iid"})
        )
    ranking = model_summary.merge(cpd_summary, on="model", how="left")
    ranking["iid_rank"] = ranking["iid_mean_macro_f1"].rank(ascending=False, method="min")
    ranking["ood_rank"] = ranking["ood_mean_macro_f1"].rank(ascending=False, method="min")
    ranking["drop_rank"] = ranking["mean_ood_drop"].rank(ascending=True, method="min")
    ranking["topology_stability_rank"] = ranking["mean_cpd_vs_iid"].rank(ascending=True, method="min")
    ranking = ranking.sort_values(["ood_rank", "topology_stability_rank", "drop_rank"])
    ranking.to_csv(report_dir / "robustness_ranking.csv", index=False, encoding="utf-8-sig")
    return ranking


def build_capacity_tables(model_summary: pd.DataFrame, cpd: pd.DataFrame, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cpd.empty:
        cpd_summary = pd.DataFrame(columns=["model", "cpd"])
    else:
        cpd_summary = (
            cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"]
            .mean()
            .rename(columns={"cpd_vs_iid_mean": "cpd"})
        )
    table = model_summary.merge(cpd_summary, on="model", how="left")
    table = table.rename(
        columns={
            "model": "Model",
            "iid_mean_macro_f1": "IID",
            "ood_mean_macro_f1": "OOD",
            "mean_ood_drop": "OOD_Drop",
            "cpd": "CPD",
        }
    )
    desired_order = ["rf", "cnn1d", "cnn1d_v2", "cnn1d_v3", "transformer", "transformer_v2"]
    table["order"] = table["Model"].map({model: index for index, model in enumerate(desired_order)})
    table = table.sort_values(["order", "Model"]).drop(columns=["order"])
    table.to_csv(report_dir / "capacity_performance_table.csv", index=False, encoding="utf-8-sig")

    rows = []
    pairs = [
        ("CNN", "cnn1d", "cnn1d_v2"),
        ("CNN-optimized", "cnn1d_v2", "cnn1d_v3"),
        ("Transformer", "transformer", "transformer_v2"),
    ]
    table_index = table.set_index("Model")
    for family, v1, v2 in pairs:
        if v1 not in table_index.index or v2 not in table_index.index:
            continue
        rows.append(
            {
                "family": family,
                "v1_model": v1,
                "v2_model": v2,
                "delta_iid": table_index.loc[v2, "IID"] - table_index.loc[v1, "IID"],
                "delta_ood": table_index.loc[v2, "OOD"] - table_index.loc[v1, "OOD"],
                "delta_ood_drop": table_index.loc[v2, "OOD_Drop"] - table_index.loc[v1, "OOD_Drop"],
                "delta_cpd": table_index.loc[v2, "CPD"] - table_index.loc[v1, "CPD"]
                if pd.notna(table_index.loc[v2, "CPD"]) and pd.notna(table_index.loc[v1, "CPD"])
                else np.nan,
            }
        )
    delta = pd.DataFrame(rows)
    delta.to_csv(report_dir / "capacity_delta.csv", index=False, encoding="utf-8-sig")
    return table, delta


def write_report(
    summaries: pd.DataFrame,
    perf: pd.DataFrame,
    model_summary: pd.DataFrame,
    cpd: pd.DataFrame,
    ranking: pd.DataFrame,
    capacity_table: pd.DataFrame,
    capacity_delta: pd.DataFrame,
    args: argparse.Namespace,
    report_dir: Path,
) -> None:
    table_index = capacity_table.set_index("Model")
    delta_index = capacity_delta.set_index("family") if not capacity_delta.empty else pd.DataFrame()

    lines = [
        "# Controlled Capacity Increase Experiment 结论\n\n",
        "## 实验目的\n\n",
        "本实验用于验证：随着模型 relation modeling capability 增强，topology drift sensitivity 是否同步增强。实验不追求 SOTA，重点比较 capacity increase 后的 IID、OOD drop、CPD 和 environment-specific fitting。\n\n",
        "## 实验设置\n\n",
        f"- 特征缓存：`{args.features}`\n",
        "- 特征集：`all_features`\n",
        "- 任务：R2/R3/R4 single-round，三组 LORO，Position R5，Jitter R6+R7\n",
        "- 对照模型：RF、CNN-v1、Transformer-v1\n",
        "- 增容模型：CNN-v2（Residual + SE）、Transformer-v2（4-layer encoder + attention pooling）\n",
        f"- 深度模型训练：epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, device={args.device}\n\n",
        "## 完整性能表\n\n",
        markdown_table(capacity_table.round(4)),
        "\n\n## Capacity Increase Delta\n\n",
        markdown_table(capacity_delta.round(4)),
        "\n\n## CPD 汇总\n\n",
        markdown_table(
            cpd.groupby("model", as_index=False)["cpd_vs_iid_mean"].mean().round(4)
            if not cpd.empty
            else pd.DataFrame(columns=["model", "cpd_vs_iid_mean"])
        ),
        "\n\n## 鲁棒性排名\n\n",
        markdown_table(ranking.round(4)),
        "\n\n## 关键问题回答\n\n",
    ]

    if "CNN" in delta_index.index:
        row = delta_index.loc["CNN"]
        lines.append(
            f"1. **CNN capacity increase 后 IID 是否提高？** CNN-v2 相比 CNN-v1 的 IID 变化为 {row['delta_iid']:.4f}，OOD 变化为 {row['delta_ood']:.4f}，OOD drop 变化为 {row['delta_ood_drop']:.4f}，CPD 变化为 {row['delta_cpd']:.4f}。\n\n"
        )
    if "Transformer" in delta_index.index:
        row = delta_index.loc["Transformer"]
        lines.append(
            f"2. **Transformer capacity increase 后 IID 是否提高？** Transformer-v2 相比 Transformer-v1 的 IID 变化为 {row['delta_iid']:.4f}，OOD 变化为 {row['delta_ood']:.4f}，OOD drop 变化为 {row['delta_ood_drop']:.4f}，CPD 变化为 {row['delta_cpd']:.4f}。\n\n"
        )

    if {"rf", "cnn1d_v2", "transformer_v2"}.issubset(table_index.index):
        rf_cpd = table_index.loc["rf", "CPD"]
        cnn2_cpd = table_index.loc["cnn1d_v2", "CPD"]
        transformer2_cpd = table_index.loc["transformer_v2", "CPD"]
        lines.append(
            f"3. **capacity increase 后 CPD 是否变大？** RF CPD={rf_cpd:.4f}，CNN-v2 CPD={cnn2_cpd:.4f}，Transformer-v2 CPD={transformer2_cpd:.4f}。需要结合 v2-v1 的 delta 判断容量增加是否同步放大 topology sensitivity。\n\n"
        )

    lines.append(
        "4. **是否出现 higher relation modeling capability -> stronger topology sensitivity？** 结论以 `capacity_delta.csv` 为准：若 v2 的 IID 上升同时 OOD drop/CPD 也上升，说明容量提升确实增强了环境相关拓扑拟合；若 IID 上升但 CPD 下降，则说明容量提升改善了表示但未放大拓扑敏感性；若 IID 未上升，则说明该 v2 设计没有形成有效的受控容量提升。\n\n"
    )

    lines.extend(
        [
            "## 输出文件\n\n",
            "- `capacity_performance_table.csv`\n",
            "- `capacity_delta.csv`\n",
            "- `performance_comparison.csv`\n",
            "- `iid_ood_model_summary.csv`\n",
            "- `cpd_comparison.csv`\n",
            "- `robustness_ranking.csv`\n",
            "- `iid_ood_degradation.png`\n",
            "- `confusion_topology_comparison.png`\n",
            "- `cpd_comparison.png`\n",
        ]
    )
    (report_dir / "DEEP_ROBUSTNESS_CONCLUSION.md").write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight CNN/Transformer IoT robustness validation")
    parser.add_argument("--config", type=Path, default=Path("code/configs/research_experiments.json"))
    parser.add_argument("--features", type=Path, default=Path("results/robust_v2/raw_all/features_raw_all_w10.csv"))
    parser.add_argument("--traditional-source", type=Path, default=Path("results/robust_v2/raw_all"))
    parser.add_argument("--deep-source", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("results/deep_robustness_validation"))
    parser.add_argument("--traditional-models", default=",".join(TRADITIONAL_MODELS))
    parser.add_argument("--copy-deep-models", default="none")
    parser.add_argument("--train-deep-models", default=",".join(DEEP_MODELS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(args.random_state)

    config = read_config(args.config)
    features = pd.read_csv(args.features)
    task_lookup = prepare_task_lookup(config)
    traditional_models = parse_model_list(args.traditional_models, TRADITIONAL_MODELS)
    copy_deep_models = parse_model_list(args.copy_deep_models, [])
    train_deep_models = parse_model_list(args.train_deep_models, DEEP_MODELS)
    selected_tasks = parse_model_list(args.tasks, TASKS)
    result_root = args.output_root
    report_dir = result_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (result_root / "raw_all").mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    environment = {
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "features": str(args.features),
        "traditional_source": str(args.traditional_source),
        "deep_source": str(args.deep_source) if args.deep_source is not None else None,
        "tasks": selected_tasks,
        "copy_deep_models": copy_deep_models,
        "train_deep_models": train_deep_models,
        "traditional_models": traditional_models,
    }
    (result_root / "environment_report.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for task_name in selected_tasks:
        task = task_lookup[task_name]
        for model_name in traditional_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model_name
            summary = load_traditional_result(model_name, task_name, args, output_dir)
            if summary is not None:
                summary["scenario"] = task_scenario(task_name)
                summaries.append(summary)

        for model_name in copy_deep_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model_name
            summary = load_deep_source_result(model_name, task_name, args, output_dir)
            if summary is not None:
                summary["scenario"] = task_scenario(task_name)
                summaries.append(summary)

        for model_name in train_deep_models:
            output_dir = result_root / "raw_all" / task_name / "all_features" / model_name
            if args.skip_existing and (output_dir / "metrics.json").exists():
                summary = load_existing_result(
                    source_root=result_root / "raw_all",
                    model_name=model_name,
                    task_name=task_name,
                    output_dir=output_dir,
                    source_label="existing_output",
                )
                if summary is not None:
                    summary["scenario"] = task_scenario(task_name)
                    summaries.append(summary)
                    print(f"Skipping existing {model_name} on {task_name}", flush=True)
                    continue

            print(f"Training {model_name} on {task_name} ({args.device})", flush=True)
            summary = evaluate_deep_task(model_name, features, task, config, args, output_dir)
            summaries.append(summary)
            print(
                f"{task_name} {model_name}: macro_f1={summary['macro_f1']:.4f}, "
                f"params={summary['parameter_count']}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(result_root / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    (result_root / "summary_metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    perf, model_summary = build_performance_tables(summary_df, report_dir)
    cpd = build_cpd_table(result_root, report_dir, selected_tasks)
    ranking = build_robustness_ranking(model_summary, cpd, report_dir)
    capacity_table, capacity_delta = build_capacity_tables(model_summary, cpd, report_dir)
    plot_degradation(perf, report_dir)
    plot_cpd(cpd, report_dir)
    plot_confusion_topology(result_root, report_dir)
    write_report(summary_df, perf, model_summary, cpd, ranking, capacity_table, capacity_delta, args, report_dir)
    print(f"Saved deep robustness validation outputs to: {result_root}")


if __name__ == "__main__":
    main()
