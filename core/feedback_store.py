# core/feedback_store.py

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from .schemas import ModerationRequest, ScoreResult, PolicyDecision

DB_PATH = Path("data/moderation_logs.db")


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def _init_db():
    conn = _get_conn()
    cur = conn.cursor()

    # 1) 基础建表（只在不存在时生效）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS moderation_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at       TEXT,
            user_id          TEXT,
            rule             TEXT,
            text             TEXT,
            p_final          REAL,
            p_deberta        REAL,
            p_qwen           REAL,
            p_gte            REAL,
            auto_action      TEXT,
            reason_code      TEXT,
            policy_version   TEXT,
            final_action     TEXT,
            final_note       TEXT,
            qwen_enabled     INTEGER,
            qwen_label       TEXT,
            qwen_explanation TEXT,
            qwen_raw         TEXT
        );
        """
    )

    # 2) 轻量迁移：确保新增列在旧库里也存在
    cur.execute("PRAGMA table_info(moderation_logs);")
    existing_cols = {row[1] for row in cur.fetchall()}  # row[1] 是列名

    expected_cols = {
        ("qwen_enabled", "INTEGER"),
        ("qwen_label", "TEXT"),
        ("qwen_explanation", "TEXT"),
        ("qwen_raw", "TEXT"),
        # 你以后再加列，就往这里追加即可
    }

    for col, col_type in expected_cols:
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE moderation_logs ADD COLUMN {col} {col_type};")

    conn.commit()
    conn.close()


# 模块导入时初始化一次
_init_db()


def log_auto_decision(
    req: ModerationRequest,
    score: ScoreResult,
    decision: PolicyDecision,
    request_id: int | None = None,
) -> int:
    """
    写入一条自动决策记录，并返回该条记录在 DB 中的 id。
    你可以忽略传入的 request_id，直接用自增主键。
    同时会把 Qwen 二审信息（若有）一起写入。
    """
    conn = _get_conn()
    cur = conn.cursor()

    created_at = datetime.now().isoformat()

    # Qwen 相关字段（可能不存在，需用 getattr）
    qwen_enabled = int(getattr(decision, "qwen_enabled", False) or 0)
    qwen_label = getattr(decision, "qwen_label", None)
    qwen_explanation = getattr(decision, "qwen_explanation", None)
    qwen_raw = getattr(decision, "qwen_raw", None)

    cur.execute(
        """
        INSERT INTO moderation_logs (
            created_at, user_id, rule, text,
            p_final, p_deberta, p_qwen, p_gte,
            auto_action, reason_code, policy_version,
            final_action, final_note,
            qwen_enabled, qwen_label, qwen_explanation, qwen_raw
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            req.user_id,
            req.rule,
            req.text,
            score.scores.get("p_final"),
            score.scores.get("p_deberta"),
            score.scores.get("p_qwen"),
            score.scores.get("p_gte"),
            decision.action,
            decision.reason_code,
            decision.policy_version,
            None,   # final_action 占位
            None,   # final_note
            qwen_enabled,
            qwen_label,
            qwen_explanation,
            qwen_raw,
        ),
    )
    db_id = cur.lastrowid
    conn.commit()
    conn.close()
    return db_id


def log_human_decision(db_id: int, final_action: str, final_note: str = "") -> None:
    """
    审核台里人工点“最终决定”之后，把这条记录补上 final_action / note。
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE moderation_logs
        SET final_action = ?, final_note = ?
        WHERE id = ?
        """,
        (final_action, final_note, db_id),
    )
    conn.commit()
    conn.close()


# ========= 审核台队列 & 详情查询接口 =========

def fetch_pending_items(
    limit: int = 200,
    action_filter: Optional[str] = None,
) -> List[Dict]:
    """
    从 moderation_logs 里拉一批“待审核”条目，供审核台左侧队列展示。

    策略：
    - 只取 final_action 为空的记录（尚未有人工最终决策）
    - 可选按 auto_action 过滤（allow/review/block）
    - 按 created_at DESC 排序
    """
    conn = _get_conn()
    cur = conn.cursor()

    base_sql = """
    SELECT
        id,
        created_at,
        user_id,
        rule,
        text,
        p_final,
        p_deberta,
        p_gte,
        auto_action AS action_model,
        final_action AS action_final
    FROM moderation_logs
    WHERE (final_action IS NULL OR final_action = '')
    """
    params: list = []

    if action_filter:
        base_sql += " AND auto_action = ?"
        params.append(action_filter)

    base_sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cur.execute(base_sql, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    result: List[Dict] = []
    for row in rows:
        item = {col: row[i] for i, col in enumerate(cols)}
        result.append(item)

    conn.close()
    return result


def fetch_log_detail(log_id: int) -> Optional[Dict]:
    """
    根据 id 获取一条 moderation_logs 的完整记录，用于右侧详情展示。

    返回的字段名尽量与审核台 UI 对齐：
    - action_model: 来自 auto_action
    - action_final: 来自 final_action
    - 以及 qwen_* 一系列字段
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
            created_at,
            user_id,
            rule,
            text,
            p_final,
            p_deberta,
            p_gte,
            auto_action AS action_model,
            final_action AS action_final,
            reason_code,
            policy_version,
            qwen_enabled,
            qwen_label,
            qwen_explanation,
            qwen_raw
        FROM moderation_logs
        WHERE id = ?
        """,
        (log_id,),
    )
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None

    cols = [d[0] for d in cur.description]
    detail = {col: row[i] for i, col in enumerate(cols)}

    # 把 qwen_enabled 从 0/1 转为 bool（方便前端判断）
    if "qwen_enabled" in detail and detail["qwen_enabled"] is not None:
        detail["qwen_enabled"] = bool(detail["qwen_enabled"])
    else:
        detail["qwen_enabled"] = False

    conn.close()
    return detail

def fetch_logs_for_dashboard(limit: int = 1000) -> List[Dict]:
    """
    拉一批日志给 Dashboard 使用。
    目前简单按时间倒序取最近 N 条。
    之后如果你想按日期区间筛选，可以再加参数。
    """
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
            created_at,
            user_id,
            rule,
            text,
            p_final,
            p_deberta,
            p_gte,
            auto_action AS action_model,
            final_action AS action_final,
            reason_code,
            policy_version,
            qwen_enabled,
            qwen_label
        FROM moderation_logs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    logs: List[Dict] = []
    for row in rows:
        item = {col: row[i] for i, col in enumerate(cols)}
        # 将 qwen_enabled 转成 bool
        if "qwen_enabled" in item and item["qwen_enabled"] is not None:
            item["qwen_enabled"] = bool(item["qwen_enabled"])
        else:
            item["qwen_enabled"] = False
        logs.append(item)

    conn.close()
    return logs
