# core/policy_pareto.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .policy_simulator import evaluate_policy, ThresholdConfig


@dataclass
class PolicyPoint:
    """
    一条策略在目标空间中的表示
    """
    t_review: float
    t_block: float

    review_rate: float
    block_rate: float
    recall: float

    metrics: Dict[str, Any]


def _dominates(a: PolicyPoint, b: PolicyPoint) -> bool:
    """
    判断策略 a 是否 Pareto-dominates 策略 b

    目标方向：
    - review_rate ↓
    - block_rate ↓
    - recall ↑
    """
    no_worse = (
        a.review_rate <= b.review_rate and
        a.block_rate  <= b.block_rate and
        a.recall      >= b.recall
    )
    strictly_better = (
        a.review_rate < b.review_rate or
        a.block_rate  < b.block_rate or
        a.recall      > b.recall
    )
    return no_worse and strictly_better


def pareto_policy_search(
    df: pd.DataFrame,
    label_col: str = "label",
    score_col: str = "p_final",
    t_review_grid: np.ndarray | None = None,
    t_block_grid: np.ndarray | None = None,
) -> List[PolicyPoint]:
    """
    在阈值空间中搜索 Pareto-optimal 策略集合

    参数
    ----
    df : pd.DataFrame
        必须包含 label_col 和 score_col
    t_review_grid : np.ndarray
        review 阈值候选集合
    t_block_grid : np.ndarray
        block 阈值候选集合（会自动保证 > t_review）

    返回
    ----
    pareto_front : List[PolicyPoint]
        非支配策略集合
    """

    if t_review_grid is None:
        t_review_grid = np.arange(0.30, 0.71, 0.02)

    if t_block_grid is None:
        t_block_grid = np.arange(0.40, 0.91, 0.02)

    candidates: List[PolicyPoint] = []

    # --------- 1) 枚举策略并评估 ---------
    for t_review in t_review_grid:
        for t_block in t_block_grid:
            if t_block <= t_review:
                continue

            cfg = ThresholdConfig(
                t_review=float(t_review),
                t_block=float(t_block),
            )

            result = evaluate_policy(
                df=df,
                label_col=label_col,
                score_col=score_col,
                cfg=cfg,
            )

            candidates.append(
                PolicyPoint(
                    t_review=float(t_review),
                    t_block=float(t_block),
                    review_rate=result["rates"]["review"],
                    block_rate=result["rates"]["block"],
                    recall=result["metrics"]["recall"],
                    metrics=result,
                )
            )

    # --------- 2) Pareto 过滤 ---------
    pareto_front: List[PolicyPoint] = []

    for p in candidates:
        dominated = False
        for q in candidates:
            if q is p:
                continue
            if _dominates(q, p):
                dominated = True
                break
        if not dominated:
            pareto_front.append(p)

    # --------- 3) 排序（方便展示） ---------
    pareto_front.sort(key=lambda x: (x.review_rate, -x.recall))

    return pareto_front
