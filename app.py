"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import sys
import uuid

# The graph nodes print progress with rich ("→", "✓"). On Windows, when Streamlit
# runs without a console, stdout falls back to cp1252 and those prints crash the run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
# The full graph (fetch → analyze → route → auto / human / escalate → commit),
# with an AuditEntry written at every node, comes from exercise 4.
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Audit / checkpoint helpers ────────────────────────────────────────────
async def list_threads(limit: int = 15) -> list[dict]:
    """Same query as `audit.replay.list_threads`, returned as rows instead of printed."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp)  AS last_event,
                   -- MAX(risk_level) would compare alphabetically ("med" > "high")
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*)        AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def load_thread(thread_id: str) -> tuple[dict | None, dict | None]:
    """Read a thread's checkpoint → (pending interrupt payload, final values)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value, None
    return None, dict(snapshot.values) if snapshot.values else None


def open_thread(thread_id: str, pr_url: str) -> None:
    payload, final = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = pr_url
    st.session_state.interrupt_payload = payload
    st.session_state.final = final


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}

with st.sidebar:
    st.header("Recent sessions")
    try:
        threads = asyncio.run(list_threads())
    except Exception as e:  # DB not created yet, etc.
        threads = []
        st.caption(f"(no audit data: {e})")
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1]
        label = f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr_short} · {t['thread_id'][:8]}"
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread_{t['thread_id']}", width="stretch",
                     type="primary" if current else "secondary"):
            open_thread(t["thread_id"], t["pr_url"])
            st.rerun()
        st.caption(f"{t['events']} events · last {str(t['last_event'])[:19]}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", width="stretch"):
        if not feedback.strip():
            st.warning("Write what should change in the Feedback box before choosing Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(f"**Q{i + 1}.** {q}", key=f"esc_q_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if not all(a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action") or "?"
    pr = st.session_state.pr_url
    if action.startswith("auto") and "failed" not in action:
        st.success(f"✓ Auto-approved (high confidence) — agent posted the review itself. "
                   f"[View comment on GitHub]({pr})")
    elif action.startswith("committed"):
        st.success(f"✓ {action} — comment posted. [View comment on GitHub]({pr})")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif "failed" in action:
        st.error(f"{action} — check GITHUB_TOKEN scope (see terminal log)")
    else:
        st.info(f"final_action = {action}")

    a = final.get("analysis")
    if a is not None:
        title = "Refined review" if final.get("escalation_answers") else "Review"
        with st.expander(f"{title} — confidence {a.confidence:.0%}", expanded=True):
            st.markdown(a.summary)
            for c in a.comments:
                st.markdown(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
            if final.get("human_feedback"):
                st.caption(f"Reviewer note: {final['human_feedback']}")


def render_audit(thread_id: str) -> None:
    events = asyncio.run(replay_events(thread_id))
    if not events:
        return
    with st.expander(f"Audit trail ({len(events)} events)"):
        st.dataframe(
            [{k: ev[k] for k in ("timestamp", "action", "confidence", "risk_level",
                                 "decision", "reviewer_id", "execution_time_ms", "reason")}
             for ev in events],
            width="stretch", hide_index=True,
        )


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    try:
        with st.spinner("Fetching PR + asking the LLM..."):
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    render_audit(st.session_state.thread_id)
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
