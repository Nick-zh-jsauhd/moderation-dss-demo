import sys
import os
from typing import Optional, List, Dict

# --------- 路径注入：确保能 import core.* ---------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../project-MIS/dss
ROOT_DIR = os.path.dirname(CURRENT_DIR)                         # .../project-MIS
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.feedback_store import (
    log_human_decision,
    fetch_pending_items,
    fetch_log_detail,
    fetch_logs_for_dashboard,
)
from core.schemas import ModerationRequest
from core.moderation_flow import handle_request
from core.scoring_service import score_comment
from core.policy_pareto import pareto_policy_search
from core.config import resolve_demo_csv

st.set_page_config(
    page_title="Moderation DSS",
    page_icon="🛡️",
    layout="wide",
)

# ================== utils ==================

def resolve_csv_path(csv_path: str) -> Path:
    p = Path(csv_path)
    if p.exists():
        return p

    # 常见兜底：data/
    p2 = DATA_DIR / csv_path
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"CSV not found: tried '{p}' and '{p2}'. "
        f"Please check dataset path."
    )

def _shorten(s: str, n: int = 48) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "..."

def _ensure_state():
    if "selected_log_id" not in st.session_state:
        st.session_state.selected_log_id = None
    if "queue_filter_action" not in st.session_state:
        st.session_state.queue_filter_action = "all"
    if "queue_limit" not in st.session_state:
        st.session_state.queue_limit = 200

def _refresh_and_select_next(prev_id: Optional[int] = None):
    pending = fetch_pending_items(
        limit=int(st.session_state.queue_limit),
        action_filter=None if st.session_state.queue_filter_action == "all" else st.session_state.queue_filter_action,
    )
    if not pending:
        st.session_state.selected_log_id = None
        return

    ids = [int(x["id"]) for x in pending if "id" in x]
    if prev_id is not None and prev_id in ids:
        idx = ids.index(prev_id)
        st.session_state.selected_log_id = ids[min(idx + 1, len(ids) - 1)]
    else:
        st.session_state.selected_log_id = ids[0]

def _seed_demo_traffic(n: int = 10):
    """
    仅用于演示：往 DB 写入一些自动决策，避免你 DB 空导致 UI 没东西可看。
    """
    import random
    demo_rules = [
        "No insulting or malicious attacks on other users.",
        "No hate speech, harassment, or violent threats.",
        "No doxxing or sharing personal information.",
        "No spam or deceptive links.",
    ]
    demo_comments = [
        "You are an idiot. Get out of here.",
        "I hope you die. Everyone like you should be eliminated.",
        "Here is his phone number: 123-456-7890. Go find him.",
        "Check this amazing offer https://example.com/free-money right now!!!",
        "I disagree with you, but let's keep it civil.",
        "This is trash and you are trash.",
        "I think your argument is weak, but no offense intended.",
        "Go kill yourself.",
        "Visit https://tinyurl.com/xyz and claim your prize.",
    ]

    for _ in range(n):
        req = ModerationRequest(
            text=random.choice(demo_comments),
            rule=random.choice(demo_rules),
            user_id="demo_user",
        )
        handle_request(req)

def _import_dataset_to_db(csv_path: str, n: int = 200):
    """
    从 train/test CSV 批量回放到 DB：
    - 每条样本走一次 handle_request -> 自动决策 -> 落库
    - 这样 Review Console / Dashboard / Policy Simulator 都会有数据
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    # 兼容字段名：尽量从常见列里找 rule/text/label
    # 你们的训练集通常包含 rule + text（或 body）
    rule_col = "rule" if "rule" in df.columns else None
    text_col = "text" if "text" in df.columns else ("body" if "body" in df.columns else None)

    if text_col is None:
        raise ValueError(f"Cannot find text column in {csv_path}. columns={list(df.columns)}")

    take = df.head(int(n)).copy()
    cnt = 0
    for _, r in take.iterrows():
        req = ModerationRequest(
            text=str(r[text_col]),
            rule=str(r[rule_col]) if rule_col else "Generic policy (dataset replay)",
            user_id="dataset_replay",
        )
        handle_request(req)
        cnt += 1
    return cnt


# ================== Page 1: Review Console ==================

def page_review_console():
    _ensure_state()

    st.title("🛡️ Moderation DSS – Review Console")
    st.caption(
        """
Enterprise workflow:
- **Pending Queue** (items without human final decision)
- **Detail panel** (scores, policy decision, optional Qwen second-opinion)
- **Human final decision** logged back to SQLite
"""
    )

    with st.sidebar:
        st.subheader("Queue Controls")

        st.session_state.queue_filter_action = st.selectbox(
            "Filter by machine action",
            options=["all", "allow", "review", "block"],
            index=["all", "allow", "review", "block"].index(st.session_state.queue_filter_action)
            if st.session_state.queue_filter_action in ["all", "allow", "review", "block"] else 0,
        )

        st.session_state.queue_limit = st.slider(
            "Queue size",
            min_value=50,
            max_value=500,
            value=int(st.session_state.queue_limit),
            step=50,
        )

        st.markdown("---")
        if st.button("🔄 Refresh queue", use_container_width=True):
            _refresh_and_select_next(prev_id=st.session_state.selected_log_id)
            st.rerun()

        with st.expander("Dataset (recommended for demo)", expanded=False):
            st.write("Replay train/test CSV into DB for realistic demo traffic.")
            csv_path = st.text_input("CSV path", value="policy_eval_data.csv")
            n = st.number_input("How many rows to import", min_value=10, max_value=5000, value=200, step=10)
            if st.button("Import dataset into DB", use_container_width=True):
                imported = _import_dataset_to_db(str(csv_path), int(n))
                _refresh_and_select_next(prev_id=None)
                st.success(f"Imported {imported} rows from {csv_path} into DB.")
                st.rerun()


    queue_col, detail_col = st.columns([1.1, 1.9])

    # ---- Left: Queue ----
    with queue_col:
        st.subheader("① Pending Queue")

        pending: List[Dict] = fetch_pending_items(
            limit=int(st.session_state.queue_limit),
            action_filter=None if st.session_state.queue_filter_action == "all" else st.session_state.queue_filter_action,
        )

        if not pending:
            st.info("No pending items (DB empty or all items already have final_action).")
            st.session_state.selected_log_id = None
        else:
            df = pd.DataFrame(pending)
            if st.session_state.selected_log_id is None or st.session_state.selected_log_id not in df["id"].tolist():
                st.session_state.selected_log_id = int(df.iloc[0]["id"])

            def _label_for_id(_id: int) -> str:
                row = df[df["id"] == _id].iloc[0].to_dict()
                action = row.get("action_model") or "?"
                p = row.get("p_final")
                p_str = f"{float(p):.3f}" if p is not None else "N/A"
                txt = _shorten(row.get("text") or "")
                return f"[{action}] p_final={p_str} · {txt}"

            options = df["id"].astype(int).tolist()
            selected_id = st.radio(
                "Pending items",
                options=options,
                format_func=_label_for_id,
                index=options.index(int(st.session_state.selected_log_id))
                if int(st.session_state.selected_log_id) in options else 0,
            )
            st.session_state.selected_log_id = int(selected_id)

            with st.expander("📋 View queue table", expanded=False):
                show_cols = ["id", "created_at", "user_id", "action_model", "action_final", "p_final", "p_deberta", "p_gte"]
                existing_cols = [c for c in show_cols if c in df.columns]
                st.dataframe(df[existing_cols], use_container_width=True, height=340)

    # ---- Right: Detail ----
    with detail_col:
        st.subheader("② Item Detail & Final Decision")

        log_id = st.session_state.selected_log_id
        if log_id is None:
            st.info("Select an item from the left queue to review.")
            return

        detail = fetch_log_detail(int(log_id))
        if not detail:
            st.error(f"Cannot find log detail for id={log_id}.")
            return

        st.markdown(f"**Log ID:** `{detail['id']}` · **Created at:** `{detail.get('created_at', '')}`")

        with st.expander("🔍 Comment & Rule", expanded=True):
            st.markdown("**User ID**")
            st.write(detail.get("user_id") or "(unknown)")

            st.markdown("**Applicable rule**")
            st.write(detail.get("rule") or "(no explicit rule)")

            st.markdown("**Comment**")
            st.write(detail.get("text") or "")

        st.markdown("### Model Scores (higher = riskier)")
        p_final = detail.get("p_final")
        p_deb = detail.get("p_deberta")
        p_gte = detail.get("p_gte")

        m1, m2, m3 = st.columns(3)
        m1.metric("Aggregated risk p_final", value=f"{p_final:.3f}" if p_final is not None else "N/A")
        m2.metric("DeBERTa head", value=f"{p_deb:.3f}" if p_deb is not None else "N/A")
        m3.metric("GTE head", value=f"{p_gte:.3f}" if p_gte is not None else "N/A")

        with st.expander("📊 Raw score fields (debug)", expanded=False):
            st.json({"p_final": p_final, "p_deberta": p_deb, "p_gte": p_gte})

        st.markdown("### Policy Decision (machine)")
        st.write(f"**Suggested action:** `{detail.get('action_model', '')}`")
        st.write(f"**Reason code:** `{detail.get('reason_code', '')}` · **Policy version:** `{detail.get('policy_version', '')}`")
        st.caption(
            "This is a baseline machine decision under the current production policy (policy_v1). "
            "Final thresholds can be adjusted via policy simulation."
        )


        st.markdown("---")
        st.subheader("③ Human Review & Feedback")
        st.markdown("Override the machine action if needed. Human decisions are logged for offline retraining and policy refinement.")

        default_option = "Keep machine suggestion"
        options = [default_option, "allow", "review", "block"]

        final_action = st.selectbox(
            "Final decision",
            options=options,
            index=0,
            key=f"final_action_{log_id}",
        )
        final_note = st.text_input("Note (optional)", key=f"final_note_{log_id}")

        if st.button("📝 Submit final decision", use_container_width=True, key=f"submit_{log_id}"):
            action_to_save = detail.get("action_model") if final_action == default_option else final_action
            if not action_to_save:
                st.error("No action to save. Please check the data.")
            else:
                log_human_decision(int(log_id), str(action_to_save), str(final_note or ""))
                st.success(f"Final decision `{action_to_save}` has been recorded.")
                _refresh_and_select_next(prev_id=int(log_id))
                st.rerun()


# ================== Page 2: Dashboard ==================

def page_dashboard():
    st.title("📊 Moderation Dashboard")
    st.caption("High-level metrics from `moderation_logs`: volume, actions, model behavior, Qwen usage.")

    logs: List[Dict] = fetch_logs_for_dashboard(limit=2000)
    if not logs:
        st.info("No logs found. Run some traffic (or seed demo items) first.")
        return

    df = pd.DataFrame(logs)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at"].dt.date

    total = len(df)
    reviewed = df["final_action"].notna().sum() if "final_action" in df.columns else 0
    qwen_rate = float(df["qwen_enabled"].mean()) if "qwen_enabled" in df.columns and total > 0 else 0.0

    c1, c2 = st.columns(2)
    c1.metric("Total items", total)
    c2.metric("Items with human final decision", reviewed)

    st.markdown("---")
    st.subheader("1) Volume & Action Trends")

    daily = df.groupby("date").size().reset_index(name="count").sort_values("date")
    st.markdown("**Daily volume**")
    st.line_chart(daily.set_index("date")["count"])

    if "auto_action" in df.columns:
        action_daily = df.groupby(["date", "auto_action"]).size().reset_index(name="count")
        pivot = action_daily.pivot(index="date", columns="auto_action", values="count").fillna(0).sort_index()
        st.markdown("**Daily action breakdown (machine)**")
        st.area_chart(pivot)

    st.markdown("---")
    st.subheader("2) Model Behavior")

    if "p_final" in df.columns:
        hist = df["p_final"].dropna()
        st.markdown("**Distribution of p_final**")
        if not hist.empty:
            import altair as alt
            hist_df = pd.DataFrame({"p_final": hist})
            chart = (
                alt.Chart(hist_df)
                .mark_bar(opacity=0.85, binSpacing=0)
                .encode(
                    alt.X("p_final:Q", bin=alt.Bin(maxbins=20), title="p_final"),
                    alt.Y("count()", title="Count"),
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No p_final values found.")

    if "auto_action" in df.columns and "p_final" in df.columns:
        st.markdown("**Average p_final by machine action**")
        agg = df.groupby("auto_action")["p_final"].mean().reset_index().rename(columns={"p_final": "avg_p_final"})
        st.bar_chart(agg.set_index("auto_action"))

    st.markdown("---")
    st.subheader("3) Rule Insights")

    if "rule" in df.columns:
        top_rules = df.groupby("rule").size().reset_index(name="count").sort_values("count", ascending=False).head(10)
        st.markdown("**Top 10 rules by volume**")
        st.table(top_rules)


def simulate_action_counts(df: pd.DataFrame, t_review: float, t_block: float, score_col: str = "p_final"):
    """
    给定一批分数 p_final，按阈值策略输出动作分布（allow/review/block）
    规则：
      - p >= t_block  -> block
      - p >= t_review -> review
      - else          -> allow
    """
    p = df[score_col].astype(float).to_numpy()

    block = (p >= t_block).sum()
    review = ((p >= t_review) & (p < t_block)).sum()
    allow = (p < t_review).sum()

    return {"allow": int(allow), "review": int(review), "block": int(block)}


# ================== Page 3: Policy Simulator ==================

def page_policy_simulator():
    st.title("🧪 Policy Simulator")
    st.caption("Simulate different thresholds on historical logs, and compare against current machine policy.")

    logs: List[Dict] = fetch_logs_for_dashboard(limit=5000)
    if not logs:
        st.info("No logs found. Run some traffic (or seed demo items) first.")
        return

    df = pd.DataFrame(logs)
    if "p_final" not in df.columns:
        st.warning("No p_final field in logs. Cannot run policy simulation.")
        return

    df = df[df["p_final"].notna()].copy()
    if df.empty:
        st.info("All logs have empty p_final. Cannot run policy simulation.")
        return

    st.subheader("1) Configure simulated policy")

    col1, col2 = st.columns(2)
    with col1:
        t_review = st.slider("Threshold for allow/review (t_review)", 0.0, 1.0, 0.40, 0.01)
    with col2:
        t_block = st.slider("Threshold for review/block (t_block)", 0.0, 1.0, 0.70, 0.01)

    if t_block <= t_review:
        st.error("t_block must be strictly greater than t_review.")
        return

    def simulate_action(p: float) -> str:
        if p < t_review:
            return "allow"
        elif p < t_block:
            return "review"
        else:
            return "block"

    df["sim_action"] = df["p_final"].apply(simulate_action)

    if "auto_action" not in df.columns:
        df["auto_action"] = None

    st.markdown("---")
    st.subheader("2) Pareto Frontier (dataset-based, for demo)")

    csv_path = st.text_input("Evaluation CSV path (with label)", value="policy_eval_data.csv", key="pareto_csv")
    n = st.number_input("Rows to evaluate", min_value=50, max_value=5000, value=300, step=50, key="pareto_n")

    import time
    import numpy as np

    if st.button("Run Pareto on dataset", use_container_width=True):
        st.info("Running Pareto search... please wait.")
        t0 = time.time()

        try:
            with st.spinner("Scoring dataset and searching Pareto frontier..."):
                dfx = pd.read_csv(resolve_csv_path(csv_path)).head(int(n)).copy()

                # label 列识别
                if "label" in dfx.columns:
                    label_col = "label"
                elif "rule_violation" in dfx.columns:
                    label_col = "rule_violation"
                else:
                    st.error(f"CSV must contain 'label' or 'rule_violation'. columns={list(dfx.columns)}")
                    st.stop()

                # 文本列识别：你这份 train.csv 是 body
                text_col = "text" if "text" in dfx.columns else ("body" if "body" in dfx.columns else None)
                rule_col = "rule" if "rule" in dfx.columns else None
                if text_col is None:
                    st.error(f"CSV must contain 'body' (or 'text'). columns={list(dfx.columns)}")
                    st.stop()

                rows = []
                for _, r in dfx.iterrows():
                    rule = str(r[rule_col]) if rule_col else "Generic moderation policy"
                    body = str(r[text_col])
                    sr = score_comment(rule=rule, body=body)
                    rows.append({"label": int(r[label_col]), "p_final": float(sr.scores["p_final"])})

                eval_df = pd.DataFrame(rows)

                # demo 快速网格（上台稳）
                t_review_grid = np.arange(0.30, 0.71, 0.05)
                t_block_grid = np.arange(0.45, 0.91, 0.05)

                pareto = pareto_policy_search(
                    eval_df,
                    label_col="label",
                    score_col="p_final",
                    t_review_grid=t_review_grid,
                    t_block_grid=t_block_grid,
                )

                if not pareto:
                    st.warning("No Pareto policies found.")
                    st.stop()

                out = pd.DataFrame([{
                    "t_review": float(p.t_review),
                    "t_block": float(p.t_block),
                    "review_rate": float(p.review_rate),
                    "block_rate": float(p.block_rate),
                    "recall": float(p.recall),
                    "precision": float(p.metrics["metrics"]["precision"]),
                    "f1": float(p.metrics["metrics"]["f1"]),
                } for p in pareto]).sort_values(["review_rate", "recall"], ascending=[True, False])

                # 关键：先 round，再做可行性过滤（避免 0.5000000001 这种“看起来相等”的行漏网）
                out["t_review"] = out["t_review"].round(4)
                out["t_block"]  = out["t_block"].round(4)

                # --- Enforce policy feasibility constraint (strict gap) ---
                eps = 1e-6
                out = out[(out["t_block"] - out["t_review"]) > eps].copy()

                if out.empty:
                    st.error("No feasible policies found after enforcing t_block > t_review.")
                    st.stop()

            st.success(f"Pareto finished. Found {len(out)} policies.")
            st.write(f"Elapsed: {time.time() - t0:.2f}s")
            st.dataframe(out, use_container_width=True, height=360)

            st.markdown("#### Manager Constraints (Acceptable Region)")
            c1, c2 = st.columns(2)
            with c1:
                max_review = st.slider("Max Review Rate (cost cap)", 0.05, 0.95, 0.50, 0.05)
            with c2:
                min_recall = st.slider("Min Recall (risk floor)", 0.50, 0.99, 0.90, 0.02)

            st.markdown("#### Baseline (Current Policy) — Conservative Default")

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                use_baseline_slider = st.checkbox("Customize baseline thresholds", value=False)
            with b2:
                t_review_base = st.slider("Baseline t_review", 0.10, 0.90, 0.70, 0.05, disabled=not use_baseline_slider)
            with b3:
                t_block_base = st.slider("Baseline t_block", 0.10, 0.99, 0.90, 0.05, disabled=not use_baseline_slider)

            # 若没开自定义，就用默认保守基线（0.70/0.90）
            if not use_baseline_slider:
                t_review_base, t_block_base = 0.70, 0.90

            # 业务可行性兜底：确保 block > review
            if t_block_base <= t_review_base:
                st.warning("Baseline must satisfy t_block > t_review. Auto-adjusting t_block.")
                t_block_base = min(0.99, t_review_base + 0.05)


            st.markdown("### Pareto Frontier Visualization (Cost vs Risk)")

            # 选推荐点
            risk_idx = out["recall"].idxmax()
            cost_idx = out["review_rate"].idxmin()
            bal_idx  = out["f1"].idxmax()

            risk_p = out.loc[risk_idx]
            cost_p = out.loc[cost_idx]
            bal_p  = out.loc[bal_idx]

            import numpy as np
            import matplotlib.pyplot as plt

            # ====== FIGURE SETUP ======
            fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

            # ====== DATA ======
            x = out["review_rate"].to_numpy()
            y = out["recall"].to_numpy()
            block = out["block_rate"].to_numpy()
            sizes = 40 + block * 450  # bubble size ∝ block rate

            # ====== ZOOM FIRST (important): set axes limits before drawing constraint shading/lines ======
            x_min = max(0.0, float(np.min(x)) - 0.02)
            x_max = min(1.0, float(np.quantile(x, 0.95)) + 0.03)
            y_min = max(0.0, float(np.quantile(y, 0.05)) - 0.03)
            y_max = min(1.0, float(np.max(y)) + 0.01)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # ====== ACCEPTABLE REGION (manager constraints): light + not dominant ======
            # 1) boundary lines (most important)
            ax.axvline(max_review, linestyle="--", linewidth=1.5, alpha=0.50)
            ax.axhline(min_recall, linestyle="--", linewidth=1.5, alpha=0.50)

            # 2) VERY light feasible box in the upper-left (optional but nice)
            #    Only shade the feasible intersection within current axes limits
            shade_x0 = x_min
            shade_x1 = min(max_review, x_max)
            shade_y0 = max(min_recall, y_min)
            shade_y1 = y_max

            if shade_x1 > shade_x0 and shade_y1 > shade_y0:
                ax.add_patch(
                    plt.Rectangle(
                        (shade_x0, shade_y0),
                        width=shade_x1 - shade_x0,
                        height=shade_y1 - shade_y0,
                        fill=True,
                        alpha=0.08,   # keep it subtle
                    )
                )
                ax.text(
                    shade_x1 - 0.01*(x_max-x_min),
                    shade_y1 - 0.06*(y_max-y_min),
                    "Acceptable\nRegion",
                    ha="right",
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    alpha=0.9,
                )

            # ====== PARETO POINTS ======
            ok = (out["review_rate"] <= max_review) & (out["recall"] >= min_recall)

            # draw infeasible points first (de-emphasized)
            ax.scatter(
                out.loc[~ok, "review_rate"],
                out.loc[~ok, "recall"],
                s=(40 + out.loc[~ok, "block_rate"].to_numpy() * 450),
                alpha=0.18,
                edgecolors="none",
            )

            # draw feasible points (slightly emphasized)
            ax.scatter(
                out.loc[ok, "review_rate"],
                out.loc[ok, "recall"],
                s=(40 + out.loc[ok, "block_rate"].to_numpy() * 450),
                alpha=0.55,
                edgecolors="none",
            )

            # ====== RECOMMENDED POINTS ======
            risk_idx = out["recall"].idxmax()
            cost_idx = out["review_rate"].idxmin()
            bal_idx  = out["f1"].idxmax()

            risk_p = out.loc[risk_idx]
            cost_p = out.loc[cost_idx]
            bal_p  = out.loc[bal_idx]

            # helper: annotate with small font (do not overcrowd)
            def label_point(row, name, dx, dy, marker="o", size=220, lw=2, alpha=0.95):
                ax.scatter(
                    [row["review_rate"]],
                    [row["recall"]],
                    s=size,
                    marker=marker,
                    facecolors="none" if marker == "o" else None,
                    linewidths=lw if marker == "o" else None,
                    alpha=alpha,
                    zorder=6,
                )
                ax.annotate(
                    name,
                    (row["review_rate"], row["recall"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=11,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    zorder=7,
                )

            # Risk-first / Cost-first as reference (lower priority)
            label_point(risk_p, "Risk-first", 10, 10, marker="o", size=240, lw=2, alpha=0.9)
            label_point(cost_p, "Cost-first", 10, -6, marker="o", size=240, lw=2, alpha=0.9)

            # Balanced as the ONLY visual focal point: STAR marker + bigger + filled
            ax.scatter(
                [bal_p["review_rate"]],
                [bal_p["recall"]],
                s=520,
                marker="*",
                alpha=0.98,
                zorder=8,
            )
            ax.annotate(
                "Balanced (Recommended)",
                (bal_p["review_rate"], bal_p["recall"]),
                textcoords="offset points",
                xytext=(10, -20),
                fontsize=12,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=9,
            )

            # ====== AXIS / TITLE ======
            ax.set_title("Policy Trade-offs: Human Review Cost vs Risk Control", pad=10)
            ax.set_xlabel("Review Rate (Human Cost)")
            ax.set_ylabel("Recall (Risk Control)")
            ax.grid(True, alpha=0.18)

            # ====== LEGEND NOTE (minimal) ======
            ax.text(
                0.99, 0.02,
                "Bubble size ∝ Block Rate",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                alpha=0.85,
            )

            st.pyplot(fig, use_container_width=True)

            st.markdown("### 3) Action distribution comparison")

            # current_policy: baseline
            curr_counts = simulate_action_counts(eval_df, t_review_base, t_block_base, score_col="p_final")

            # simulated_policy: 推荐策略（用 bal_p 或你选中的某个点）
            sim_counts = simulate_action_counts(eval_df, float(bal_p["t_review"]), float(bal_p["t_block"]), score_col="p_final")

            dist_df = pd.DataFrame(
                {
                    "current_policy": curr_counts,
                    "simulated_policy": sim_counts,
                }
            )

            # 顺序更符合业务阅读习惯
            dist_df = dist_df.loc[["allow", "review", "block"]]

            st.dataframe(dist_df, use_container_width=True)

            # 可选：同时给出百分比（更好讲）
            total = len(eval_df)
            pct_df = (dist_df / total).rename(columns={"current_policy": "current_policy_%", "simulated_policy": "simulated_policy_%"})
            pct_df = (pct_df * 100).round(1)
            st.caption(f"Counts are based on N={total} samples. Percentages shown below.")
            st.dataframe(pct_df, use_container_width=True)

            st.markdown("#### Action distribution (counts)")

            import numpy as np
            import matplotlib.pyplot as plt

            labels = ["allow", "review", "block"]
            curr_vals = [curr_counts[k] for k in labels]
            sim_vals  = [sim_counts[k] for k in labels]

            x = np.arange(len(labels))
            width = 0.36

            fig2, ax2 = plt.subplots(figsize=(7.5, 4.2), dpi=140)

            ax2.bar(x - width/2, curr_vals, width, label="Current policy", alpha=0.75)
            ax2.bar(x + width/2, sim_vals,  width, label="Simulated policy", alpha=0.85)

            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel("Number of items")
            ax2.set_title("Action distribution comparison (counts)")
            ax2.legend()

            ax2.grid(axis="y", alpha=0.25)

            st.pyplot(fig2, use_container_width=True)


            st.markdown("---")
            st.subheader("Balanced Strategy (Recommended)")
            st.caption(
                "Balanced strategy is selected from the Pareto frontier by maximizing F1 "
                "under operational feasibility constraints."
            )

            # 1) 先做字段校验，防止显示出“0.50/0.50”“review=0.000”这种假象
            required_cols = ["t_review", "t_block", "review_rate", "block_rate", "recall", "precision", "f1"]
            missing = [c for c in required_cols if c not in out.columns]
            if missing:
                st.error(f"Manager card missing columns: {missing}. Available: {list(out.columns)}")
                st.stop()

            # 2) 明确：Balanced = F1 最大（你也可以换成 knee point 或 recall-λ*review）
            bal_idx = out["f1"].idxmax()
            bal = out.loc[bal_idx].copy()
            # st.write("DEBUG raw thresholds:", repr(float(bal["t_review"])), repr(float(bal["t_block"])), "diff=", float(bal["t_block"])-float(bal["t_review"]))

            # 3) 业务一致性校验：t_block 应该 > t_review
            if float(bal["t_block"]) <= float(bal["t_review"]):
                st.warning(
                    f"Unexpected thresholds (t_block <= t_review): "
                    f"{bal['t_review']:.2f}, {bal['t_block']:.2f}. "
                    f"Check out dataframe construction."
                )

            # 4) 卡片布局：左侧参数与一句话，右侧 KPI
            left, right = st.columns([1.05, 1.2])

            with left:
                st.markdown("**Policy Parameters**")
                a, b = st.columns(2)
                a.metric("t_review", f"{float(bal['t_review']):.3f}")
                b.metric("t_block",  f"{float(bal['t_block']):.3f}")

                st.markdown("**One-line recommendation**")
                st.info(
                    f"Recommended policy: set review threshold at {bal['t_review']:.2f} "
                    f"and block threshold at {bal['t_block']:.2f}, "
                    f"achieving {bal['recall']:.1%} risk recall "
                    f"with only {bal['review_rate']:.1%} human review cost."
                )

            with right:
                st.markdown("**Operational KPIs**")

                def kpi_row(name: str, value: float):
                    c1, c2 = st.columns([0.45, 0.55])
                    c1.metric(name, f"{value*100:.1f}%")
                    c2.progress(min(max(value, 0.0), 1.0))

                # 注意：review/block 是“成本/动作比例”，也在 0-1，但语义不是“越大越好”
                kpi_row("Review Rate (Human Cost)", float(bal["review_rate"]))
                kpi_row("Block Rate (Auto Blocking)", float(bal["block_rate"]))

            st.caption("Balanced strategy is selected from the Pareto-optimal set by maximizing F1.")

        except Exception as e:
            st.exception(e)




# ================== Main ==================

def main():
    with st.sidebar:
        st.header("🛡️ Moderation DSS")
        page = st.radio("View", options=["Review Console", "Dashboard", "Policy Simulator"])

    if page == "Review Console":
        page_review_console()
    elif page == "Dashboard":
        page_dashboard()
    else:
        page_policy_simulator()

if __name__ == "__main__":
    main()
