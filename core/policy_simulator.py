# core/policy_simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class ThresholdConfig:
    """策略阈值配置"""
    t_block: float = 0.8   # >= t_block → block
    t_review: float = 0.5  # >= t_review → review, 否则 allow


def apply_threshold_policy(
    p_final: np.ndarray,
    cfg: ThresholdConfig,
) -> np.ndarray:
    """
    输入：p_final 数组
    输出：action 数组（'allow'/'review'/'block'）
    """
    actions = np.full_like(p_final, "", dtype=object)

    actions[p_final >= cfg.t_block] = "block"
    mid_mask = (p_final >= cfg.t_review) & (p_final < cfg.t_block)
    actions[mid_mask] = "review"
    actions[p_final < cfg.t_review] = "allow"

    return actions


def evaluate_policy(
    df: pd.DataFrame,
    label_col: str = "label",
    score_col: str = "p_final",
    cfg: ThresholdConfig | None = None,
) -> Dict[str, Any]:
    """
    在带标签的数据集上评估某一套阈值策略。
    label: 1 = 违规, 0 = 不违规
    """
    if cfg is None:
        cfg = ThresholdConfig()

    if label_col not in df.columns:
        raise ValueError(f"DataFrame 中找不到标签列: {label_col}")
    if score_col not in df.columns:
        raise ValueError(f"DataFrame 中找不到分数字段: {score_col}")

    y_true = df[label_col].to_numpy().astype(int)
    p_final = df[score_col].to_numpy().astype(float)

    actions = apply_threshold_policy(p_final, cfg)

    # 把动作映射成“预测是否违规”：block/review → 1, allow → 0
    y_pred = np.where(actions == "allow", 0, 1)

    # 分类指标
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, p_final)
    except Exception:
        auc = float("nan")

    # confusion matrix: [[tn, fp], [fn, tp]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    n = len(df)
    allow_rate = float(np.mean(actions == "allow"))
    review_rate = float(np.mean(actions == "review"))
    block_rate = float(np.mean(actions == "block"))

    return {
        "thresholds": {
            "t_block": cfg.t_block,
            "t_review": cfg.t_review,
        },
        "n_samples": n,
        "rates": {
            "allow": allow_rate,
            "review": review_rate,
            "block": block_rate,
        },
        "cm": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        },
    }
