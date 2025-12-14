# core/policy_engine.py
from pathlib import Path
import yaml
from .schemas import ScoreResult, PolicyDecision

_POLICY = None

GRAY_LOW = 0.55
GRAY_HIGH = 0.7

def is_gray_zone(p_final: float, action_model: str) -> bool:
    # 按需要调整
    if action_model == "review":
        return True
    return GRAY_LOW <= p_final <= GRAY_HIGH

def load_policy():
    global _POLICY
    if _POLICY is None:
        path = Path("policies/policy_v1.yaml")
        with path.open("r", encoding="utf-8") as f:
            _POLICY = yaml.safe_load(f)

def apply_policy(score: ScoreResult) -> PolicyDecision:
    load_policy()
    p = score.scores.get("p_final", 0.0)

    t_block = _POLICY["thresholds"]["block"]
    t_review = _POLICY["thresholds"]["review"]

    if p >= t_block:
        action_key = "block"
        reason = "VIOLATION_HIGH"
    elif p >= t_review:
        action_key = "review"
        reason = "VIOLATION_BORDERLINE"
    else:
        action_key = "allow"
        reason = "SAFE"

    action = _POLICY["action_mapping"][action_key]
    return PolicyDecision(
        action=action,
        reason_code=reason,
        policy_version=_POLICY["id"],
    )
