# core/schemas.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModerationRequest:
    text: str
    rule: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[str] = None  # 前后文、帖子标题等

@dataclass
class ScoreResult:
    scores: Dict[str, float]       # {"p_final": 0.8, "p_qwen": 0.9, ...}
    model_version: str

@dataclass
class PolicyDecision:
    action: str                    # "allow" / "review" / "block"
    reason_code: str               # "SAFE" / "TOXIC_BORDERLINE" 等
    policy_version: str
