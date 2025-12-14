from dataclasses import dataclass
from .schemas import ScoreResult
from .ensemble.light_ml_head import base_head, ext_strict_head


@dataclass
class EnsembleConfig:
    w_deberta: float = 0.7
    w_gte: float = 0.3


ENSEMBLE_CFG = EnsembleConfig()


def score_comment(
    rule: str | None,
    body: str | None,
    subreddit: str | None = None,
    cfg: EnsembleConfig = ENSEMBLE_CFG,
) -> ScoreResult:
    """线上评分接口：返回最终分数 + 各子模型分数"""

    rule_str = rule or ""
    body_str = body or ""

    # 1) Base 轻量头（映射为 p_deberta）
    p_deb = base_head().predict_proba([rule_str], [body_str])[0]

    # 2) Ext strict 轻量头（映射为 p_gte）
    p_gte = ext_strict_head().predict_proba([rule_str], [body_str])[0]

    # 3) 线性融合
    w = cfg
    w_sum = w.w_deberta + w.w_gte
    p_final = (
        w.w_deberta * p_deb +
        w.w_gte     * p_gte
    ) / (w_sum if w_sum > 0 else 1.0)

    return ScoreResult(
        scores={
            "p_final":   p_final,
            "p_deberta": p_deb,
            "p_gte":     p_gte,
        },
        model_version="ensemble_v1_lightml_base_extstrict",
    )
