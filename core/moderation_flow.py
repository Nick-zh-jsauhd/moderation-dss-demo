# core/moderation_flow.py
from __future__ import annotations

from .schemas import ModerationRequest, ScoreResult, PolicyDecision
from . import scoring_service, policy_engine
from .feedback_store import log_auto_decision
ENABLE_QWEN = False  # demo/本地未部署时关闭


# --------- 灰区判定的一组超参数（可以之后挪到 config.py） ----------
# 这里的逻辑是：
#   - p_final 落在 [GRAY_LOW, GRAY_HIGH] 之间，或者
#   - 策略层本身给出的 action 是 "review"
# 就认为是“灰区”，触发 Qwen 二次裁决 + 解释。
GRAY_LOW = 0.55
GRAY_HIGH = 0.70


def _get_p_final(scores: ScoreResult) -> float | None:
    """
    从 ScoreResult.scores 里拿出 p_final。
    ScoreResult 是 dataclass，结构为：
        scores: Dict[str, float]  # {"p_final": 0.8, "p_deberta": 0.9, ...}
    """
    score_dict = scores.scores
    if isinstance(score_dict, dict) and "p_final" in score_dict:
        return float(score_dict["p_final"])
    return None


def _is_gray_zone(scores: ScoreResult, decision: PolicyDecision) -> bool:
    """
    灰区判定规则：
    - action == "review" 直接认为是灰区
    - 否则看 p_final 是否落在 [GRAY_LOW, GRAY_HIGH] 区间
    """
    action = decision.action
    if action and action.lower() == "review":
        return True

    p_final = _get_p_final(scores)
    if p_final is None:
        return False

    return GRAY_LOW <= p_final <= GRAY_HIGH


def handle_request(req: ModerationRequest):
    """
    处理一次审核请求完整流程：

    1）调用模型打分（DeBERTa + GTE ensemble）
    2）调用策略引擎（policy_engine.apply_policy）
    3）如处在“灰区”，调用 Qwen 做二次裁决 / 解释
    4）写入 SQLite 日志（log_auto_decision）
    5）返回 (db_id, ScoreResult, PolicyDecision)

    说明：
    - 返回值保持原样，只是给 PolicyDecision 实例动态挂了几个 qwen_* 字段：
        - decision.qwen_enabled: bool
        - decision.qwen_raw: str | None          （原始 JSON 字符串）
        - decision.qwen_label: str | None        （"yes"/"no"/"unknown"）
        - decision.qwen_explanation: str | None  （解析后的中文解释）
    """
    # 1) 模型打分：ScoreResult（里面包含 p_final / p_deberta / p_gte 等）
    scores: ScoreResult = scoring_service.score_comment(
        rule=req.rule,
        body=req.text,
        subreddit=None,  # 目前没有 subreddit，可后续扩展
    )

    # 2) 策略层决策：PolicyDecision（action / reason_code / policy_version）
    decision: PolicyDecision = policy_engine.apply_policy(scores)

    # ------- 3) 灰区触发 Qwen 二次裁决 / 规则解释 -------
    qwen_enabled = False
    qwen_raw = None
    qwen_label = None
    qwen_explanation = None

    if ENABLE_QWEN and _is_gray_zone(scores, decision):
        qwen_enabled = True
        try:
            from .llm.qwen_client import second_opinion  # 延迟导入，避免未部署时崩
            qwen_result = second_opinion(
                rule=req.rule,
                comment=req.text,
                p_final=_get_p_final(scores),
                action=decision.action,
            )
            if isinstance(qwen_result, dict):
                qwen_raw = str(qwen_result.get("raw", ""))
                qwen_label = qwen_result.get("label")
                qwen_explanation = qwen_result.get("explanation")

        except Exception as e:
            # demo 阶段不要因为 LLM 报错整个流程崩掉
            qwen_raw = f"[Qwen error] {e!r}"
            qwen_label = None
            qwen_explanation = None

    # 把 Qwen 相关信息挂到 PolicyDecision 上，前端 / dashboard 可以直接读取
    # dataclass 本质上也是普通 Python 对象，可以动态加属性
    decision.qwen_enabled = qwen_enabled
    decision.qwen_raw = qwen_raw
    decision.qwen_label = qwen_label
    decision.qwen_explanation = qwen_explanation

    # 4) 写入自动决策日志（目前 DB 里先只记核心字段；
    #    后续你可以扩展表结构把 qwen_* 也写进去）
    db_id: int = log_auto_decision(
        req=req,
        score=scores,
        decision=decision,
    )

    # 5) 返回给前端 / 调用方
    return db_id, scores, decision
