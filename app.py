"""
app.py
ApplyX Streamlit application.

UI is intentionally downstream of the authoritative orchestrator/state machine:
preview analysis is informational; application lifecycle, draft versions,
approval and submission remain orchestrator-controlled.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import sqlite3
import time
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from core.agent_tools import reset_orchestrator, set_orchestrator, submit_application
from core.decision_engine import decide_from_match_analysis
from core.matching_engine import evaluate_candidate_match
from core.orchestrator import ApplicationOrchestrator, DEMO_MODE
from core.state_machine import ApplicationStatus
from core.utils import calculate_profile_completeness, hash_password, verify_password
from diagnostics.run_benchmark import has_real_credentials, run_agent_workflow_benchmark, run_deterministic_benchmark

st.set_page_config(
    page_title="ApplyX — Autonomous Recruitment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
STYLE_CANDIDATES = [ROOT / "styles.css", ROOT / "style.css"]
for style_path in STYLE_CANDIDATES:
    if style_path.exists():
        st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
        break

DB_FILE_PATH = str(ROOT / "agent_applications.db")
AGENT_MODE = os.getenv("APPLYX_AGENT_MODE", "REAL").strip().upper()


def init_database(db_path: str = DB_FILE_PATH) -> None:
    ApplicationOrchestrator(db_path=db_path)._ensure_schema_initialized()


init_database()
orch = ApplicationOrchestrator(db_path=DB_FILE_PATH)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user_id" not in st.session_state:
    st.session_state.current_user_id = None
if "current_username" not in st.session_state:
    st.session_state.current_username = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "opportunities"
if "selected_opp_for_apply" not in st.session_state:
    st.session_state.selected_opp_for_apply = None
if "live_app_id" not in st.session_state:
    st.session_state.live_app_id = None
if "live_app_stage" not in st.session_state:
    st.session_state.live_app_stage = "IDLE"
if "live_app_res" not in st.session_state:
    st.session_state.live_app_res = None


def _new_user_id() -> str:
    return f"USER-{secrets.token_hex(6).upper()}"


def _seed_opportunities() -> None:
    sample = [
        {
            "id": "OPP-EXTRACT-01",
            "title": "Machine Learning Research Intern",
            "organization": "DeepMind Core Labs",
            "type": "Internship",
            "hard_requirements": ["Python", "PyTorch", "Work authorization required"],
            "preferred_requirements": ["Linear Algebra", "Machine Learning"],
            "source_url": "https://careers.google.com",
        },
        {
            "id": "OPP-EXTRACT-02",
            "title": "Autonomous Systems Software Fellow",
            "organization": "ScaleOps Cloud",
            "type": "Internship",
            "hard_requirements": ["Python", "SQL / Relational Databases", "Work authorization required"],
            "preferred_requirements": ["Playwright", "Git"],
            "source_url": "https://scaleops.com/careers",
        },
        {
            "id": "OPP-EXTRACT-03",
            "title": "National STEM Excellence Scholarship",
            "organization": "Higher Education Trust",
            "type": "Scholarship",
            "hard_requirements": ["Minimum CGPA 7.5 cut-off", "Enrolled in B.Tech/STEM"],
            "preferred_requirements": ["Technical Projects"],
            "source_url": "https://stemtrust.org/grants",
        },
        {
            "id": "OPP-EXTRACT-04",
            "title": "Lead Artificial Intelligence Scientist",
            "organization": "Frontier AI Labs",
            "type": "Full-Time",
            "hard_requirements": ["PhD in Computer Science or Artificial Intelligence required"],
            "preferred_requirements": ["Published at NeurIPS"],
            "source_url": "https://frontierai.org/careers",
        },
    ]
    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    try:
        for opp in sample:
            conn.execute(
                """
                INSERT OR REPLACE INTO opportunities (id,title,organization,type,requirements,source_url,created_at)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    opp["id"], opp["title"], opp["organization"], opp["type"],
                    json.dumps({"hard": opp["hard_requirements"], "preferred": opp["preferred_requirements"]}),
                    opp["source_url"], datetime.now().isoformat(),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _safe_json(raw: Any, default: Any) -> Any:
    try:
        value = json.loads(raw or "")
        return value
    except (TypeError, json.JSONDecodeError):
        return default


def _read_draft_bundle(application_id: str) -> tuple[list[dict[str, Any]], Optional[sqlite3.Row]]:
    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        drafts = [dict(r) for r in conn.execute(
            "SELECT id,version,fields,answers,documents,created_at FROM application_drafts WHERE application_id=? ORDER BY version",
            (application_id,),
        ).fetchall()]
        app = conn.execute(
            "SELECT id,user_id,opportunity_id,current_stage,approved_version,approved_draft_id,submitted_version,confirmation_id,updated_at FROM applications WHERE id=?",
            (application_id,),
        ).fetchone()
        return drafts, app
    finally:
        conn.close()


def _opportunity_for_id(opportunity_id: str) -> Optional[dict[str, Any]]:
    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    try:
        row = conn.execute(
            "SELECT id,title,organization,type,requirements,source_url FROM opportunities WHERE id=?",
            (opportunity_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    req = _safe_json(row[4], {})
    return {
        "id": row[0],
        "title": row[1],
        "organization": row[2],
        "type": row[3],
        "hard_requirements": req.get("hard", []) if isinstance(req, dict) else [],
        "preferred_requirements": req.get("preferred", []) if isinstance(req, dict) else [],
        "source_url": row[5],
    }


def _format_experience(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                company = item.get("company") or ""
                role = item.get("role") or item.get("title") or ""
                period = item.get("period") or ""
                line = " · ".join(str(x) for x in (role, company, period) if str(x).strip())
                if line:
                    parts.append(line)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _format_education(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                degree = item.get("degree") or ""
                college = item.get("college") or item.get("institution") or ""
                year = item.get("graduation_year") or item.get("year") or ""
                line = " · ".join(str(x) for x in (degree, college, year) if str(x).strip())
                if line:
                    parts.append(line)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _display_draft(draft: dict[str, Any], key_prefix: str, allow_edit: bool = True) -> Optional[dict[str, Any]]:
    fields = _safe_json(draft.get("fields"), {})
    answers = _safe_json(draft.get("answers"), {})
    documents = _safe_json(draft.get("documents"), [])

    st.subheader(f"📄 Application Form — Draft V{draft['version']}")
    st.caption("This is the exact immutable draft the agent prepared. The selected version is what the human reviewer will approve.")

    st.markdown("#### Candidate / Form Fields")
    field_rows = []
    for name, payload in fields.items():
        if isinstance(payload, dict):
            value = payload.get("value")
            verified = payload.get("verified")
            source = payload.get("source", "")
        else:
            value = payload
            verified = None
            source = ""
        if isinstance(value, (list, dict)):
            if name == "work_experience":
                shown = _format_experience(value)
            elif name == "education_history":
                shown = _format_education(value)
            else:
                shown = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            shown = "" if value is None else str(value)
        field_rows.append({
            "Field": name.replace("_", " ").title(),
            "Value": shown or "—",
            "Verified": "✅" if verified else "⚪" if verified is not None else "",
            "Source": source,
        })
    if field_rows:
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Agent-Filled Answers")
    if answers:
        for name, payload in answers.items():
            value = payload.get("value", "") if isinstance(payload, dict) else str(payload)
            st.text_area(
                name.replace("_", " ").title(),
                value=str(value),
                height=120,
                disabled=not allow_edit,
                key=f"{key_prefix}_answer_{name}",
            )
    else:
        st.info("No free-text answers were generated for this draft.")

    st.markdown("#### 📎 Documents")
    if documents:
        for doc in documents:
            if isinstance(doc, dict):
                st.write(f"• **{doc.get('name', 'Document')}** — `{doc.get('path', '')}`")
    else:
        st.info("No supporting documents attached to this version.")

    if not allow_edit:
        return None
    return {"fields": fields, "answers": answers, "documents": documents}


def _render_approval_queue() -> None:
    st.markdown("<h2 class='main-header-title'>🛡 Human Approval Queue</h2>", unsafe_allow_html=True)
    st.write("Applications prepared by ApplyX are held here until you review the exact form content and explicitly approve a version.")

    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT a.id, a.opportunity_id, a.current_stage, a.approved_version, a.approved_draft_id,
                   a.updated_at, o.title, o.organization, o.type, o.requirements, o.source_url
            FROM applications a
            LEFT JOIN opportunities o ON o.id = a.opportunity_id
            WHERE a.user_id = ? AND a.current_stage IN (?, ?)
            ORDER BY a.updated_at DESC, a.rowid DESC
            """,
            (user_id, ApplicationStatus.AWAITING_APPROVAL.value, ApplicationStatus.APPROVED.value),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        st.success("✅ No applications are waiting for human approval.")
        st.info("When ApplyX prepares a draft, it will appear here before anything can be submitted.")
        return

    st.metric("Waiting for approval", len(rows))

    for row in rows:
        with st.container(border=True):
            st.markdown(f"### {row['title']}")
            st.write(f"🏢 **{row['organization']}** · `{row['type']}`")
            st.caption(f"Application ID: `{row['id']}`")

            req = _safe_json(row["requirements"], {})
            hard = req.get("hard", []) if isinstance(req, dict) else []
            preferred = req.get("preferred", []) if isinstance(req, dict) else []

            req_col1, req_col2 = st.columns(2)
            with req_col1:
                st.markdown("**Hard requirements**")
                for item in hard or ["None specified"]:
                    st.write(f"🔴 {item}")
            with req_col2:
                st.markdown("**Preferred requirements**")
                for item in preferred or ["None specified"]:
                    st.write(f"🟡 {item}")

            drafts, app_row = _read_draft_bundle(row["id"])
            if not drafts:
                st.error("Draft record is missing; approval is disabled until the draft exists.")
                continue

            latest = drafts[-1]
            versions = [int(d["version"]) for d in drafts]
            selected_version = st.selectbox(
                "Select draft version to review",
                versions,
                index=len(versions) - 1,
                key=f"queue_version_{row['id']}",
            )
            selected = next(d for d in drafts if int(d["version"]) == selected_version)
            _display_draft(selected, f"queue_{row['id']}_{selected_version}", allow_edit=False)

            with st.expander("✏️ Edit this version before approval", expanded=False):
                answers = _safe_json(selected.get("answers"), {})
                fields = _safe_json(selected.get("fields"), {})
                edit_answers: dict[str, Any] = {}
                for name, payload in answers.items():
                    current = payload.get("value", "") if isinstance(payload, dict) else str(payload)
                    updated = st.text_area(
                        name.replace("_", " ").title(),
                        value=str(current),
                        key=f"queue_edit_{row['id']}_{selected_version}_{name}",
                    )
                    edit_answers[name] = {
                        **(payload if isinstance(payload, dict) else {}),
                        "value": updated.strip(),
                        "source": "human_edited_before_approval",
                    }
                if st.button("💾 Save as New Version", key=f"queue_save_{row['id']}_{selected_version}"):
                    revised = orch.create_revised_draft(
                        row["id"],
                        updated_fields=fields,
                        updated_answers=edit_answers,
                    )
                    st.success(f"Created Draft V{revised['version']}. Previous versions remain immutable.")
                    st.rerun()

            already_approved = (
                app_row
                and app_row["current_stage"] == ApplicationStatus.APPROVED.value
                and int(app_row["approved_version"] or 0) == selected_version
            )
            if already_approved:
                st.success(f"✅ Human approved Draft V{selected_version}. The exact approved version is still protected by the final policy gate.")
                if st.button(
                    f"🌐 Dispatch Approved V{selected_version}",
                    key=f"queue_dispatch_{row['id']}_{selected_version}",
                    type="primary",
                    use_container_width=True,
                ):
                    token = set_orchestrator(orch)
                    try:
                        submission = asyncio.run(submit_application(row["id"], selected_version))
                    finally:
                        reset_orchestrator(token)
                    if submission.get("success"):
                        st.success(f"✅ Submitted successfully. Confirmation: `{submission.get('confirmation_id')}`")
                        st.rerun()
                    else:
                        st.error(submission.get("message") or submission.get("error") or "Submission failed.")
            else:
                st.warning("Nothing will be submitted until you explicitly approve this exact draft version.")
                if st.button(
                    f"👍 Approve Draft V{selected_version}",
                    key=f"queue_approve_{row['id']}_{selected_version}",
                    type="primary",
                    use_container_width=True,
                ):
                    approval = orch.approve_draft_version(row["id"], selected_version)
                    if approval.get("success"):
                        st.success(f"Human approval recorded for Draft V{selected_version}.")
                        st.rerun()
                    else:
                        st.error(approval.get("error", "Approval failed."))

    st.divider()
    st.caption("After approval, return here or open the application's submission step. The policy gate still requires the exact approved version.")


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("<h1 class='main-header-title'>🤖 ApplyX</h1>", unsafe_allow_html=True)
        st.markdown("### Autonomous Recruitment Agent & Multi-Dimensional Matching Engine")
        st.write("ApplyX evaluates opportunities, requests missing evidence when necessary, creates immutable application drafts, enforces human approval, and dispatches the approved version through a deterministic browser boundary.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Lifecycle", "SQLite + FSM")
        m2.metric("Agent Mode", AGENT_MODE)
        m3.metric("Browser", "DEMO" if DEMO_MODE else "PLAYWRIGHT")

    with right:
        with st.container(border=True):
            st.subheader("Welcome to ApplyX")
            login_tab, register_tab = st.tabs(["🔑 Sign In", "📝 Create Account"])

            with login_tab:
                with st.form("login_form"):
                    username = st.text_input("Username").strip()
                    password = st.text_input("Password", type="password")
                    submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                    if submit_login:
                        if not username or not password:
                            st.error("Username and password are required.")
                        else:
                            conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
                            try:
                                row = conn.execute("SELECT user_id,password_hash FROM auth_users WHERE username=?", (username,)).fetchone()
                            finally:
                                conn.close()
                            if row and verify_password(password, row[1]):
                                st.session_state.authenticated = True
                                st.session_state.current_user_id = row[0]
                                st.session_state.current_username = username
                                st.session_state.current_page = "opportunities"
                                st.rerun()
                            else:
                                st.error("Invalid username or password.")

            with register_tab:
                with st.form("register_form"):
                    new_username = st.text_input("Choose Username").strip()
                    new_password = st.text_input("Choose Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    create = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                    if create:
                        if not new_username or not new_password:
                            st.error("All fields are required.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            user_id = _new_user_id()
                            try:
                                conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
                                conn.execute(
                                    "INSERT INTO auth_users(user_id,username,password_hash,created_at) VALUES (?,?,?,?)",
                                    (user_id, new_username, hash_password(new_password), datetime.now().isoformat()),
                                )
                                conn.commit()
                                conn.close()
                                orch.fetch_user_profile(user_id)
                                st.session_state.authenticated = True
                                st.session_state.current_user_id = user_id
                                st.session_state.current_username = new_username
                                st.session_state.current_page = "profile"
                                st.rerun()
                            except sqlite3.IntegrityError:
                                st.error("Username already exists.")
    st.stop()


user_id = st.session_state.current_user_id
username = st.session_state.current_username
profile = orch.fetch_user_profile(user_id)
completeness = calculate_profile_completeness(profile)

with st.sidebar:
    st.markdown("### 🤖 ApplyX Control Hub")
    st.markdown(f"👤 **{username}**")
    st.caption(f"ID: `{user_id}`")
    st.progress(completeness["total"] / 100, text=f"Profile completeness: {completeness['total']}%")
    st.caption(f"Agent mode: `{AGENT_MODE}`")
    st.caption(f"Browser mode: `{'DEMO' if DEMO_MODE else 'REAL'}`")
    st.divider()
    # Approval queue count is calculated from persisted application state.
    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    try:
        approval_count = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE user_id=? AND current_stage=?",
            (user_id, ApplicationStatus.AWAITING_APPROVAL.value),
        ).fetchone()[0]
    finally:
        conn.close()
    approval_label = f"🛡 Approval Queue ({approval_count})" if approval_count else "🛡 Approval Queue"
    pages = [("🌐 Opportunities", "opportunities"), ("⚡ Live Automation", "automation"), (approval_label, "approval_queue"), ("📜 History", "history"), ("📊 Benchmark", "benchmark"), ("⚙️ Profile", "profile")]
    for label, page in pages:
        if st.button(label, key=f"nav_{page}", use_container_width=True, type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user_id = None
        st.session_state.current_username = None
        st.rerun()


# -----------------------------------------------------------------------------
# Opportunities
# -----------------------------------------------------------------------------
if st.session_state.current_page == "opportunities":
    _seed_opportunities()
    st.markdown("<h2 class='main-header-title'>🌐 Opportunity Discovery</h2>", unsafe_allow_html=True)

    with st.expander("➕ Add Custom Opportunity", expanded=False):
        with st.form("custom_opp"):
            a, b = st.columns(2)
            with a:
                title = st.text_input("Opportunity Title")
                organization = st.text_input("Organization")
                opp_type = st.selectbox("Type", ["Internship", "Full-Time", "Scholarship", "Fellowship", "Grant"])
            with b:
                hard_text = st.text_area("Hard Requirements (comma-separated)")
                pref_text = st.text_area("Preferred Requirements (comma-separated)")
                source_url = st.text_input("Source URL")
            create_opp = st.form_submit_button("Add Opportunity", type="primary")
            if create_opp:
                if not title.strip() or not organization.strip():
                    st.error("Title and organization are required.")
                else:
                    opp_id = f"OPP-CUSTOM-{secrets.token_hex(5).upper()}"
                    hard = [x.strip() for x in hard_text.split(",") if x.strip()]
                    preferred = [x.strip() for x in pref_text.split(",") if x.strip()]
                    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
                    conn.execute(
                        "INSERT INTO opportunities(id,title,organization,type,requirements,source_url,created_at) VALUES(?,?,?,?,?,?,?)",
                        (opp_id, title.strip(), organization.strip(), opp_type, json.dumps({"hard": hard, "preferred": preferred}), source_url.strip() or None, datetime.now().isoformat()),
                    )
                    conn.commit(); conn.close()
                    st.success("Opportunity added.")
                    st.rerun()

    q1, q2, q3 = st.columns([2, 1, 1])
    with q1:
        query = st.text_input("🔍 Search", placeholder="title or organization").strip().lower()
    with q2:
        type_filter = st.selectbox("Type", ["All Types", "Internship", "Full-Time", "Scholarship", "Fellowship", "Grant"])
    with q3:
        min_fit = st.slider("Minimum Fit", 0, 100, 0, 5)

    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    rows = conn.execute("SELECT id,title,organization,type,requirements,source_url FROM opportunities ORDER BY rowid DESC").fetchall()
    conn.close()

    for opp_id, opp_title, org, opp_type, req_json, source_url in rows:
        try:
            req = json.loads(req_json or "{}")
        except json.JSONDecodeError:
            req = {}
        hard = req.get("hard", []) if isinstance(req, dict) else []
        preferred = req.get("preferred", []) if isinstance(req, dict) else []
        if query and not any(query in str(v).lower() for v in (opp_title, org, opp_type, " ".join(hard + preferred))):
            continue
        if type_filter != "All Types" and opp_type != type_filter:
            continue
        raw_opp = {"id": opp_id, "title": opp_title, "organization": org, "type": opp_type, "hard_requirements": hard, "preferred_requirements": preferred, "source_url": source_url}
        canonical = orch.normalize_opportunity(raw_opp)
        analysis = evaluate_candidate_match(profile, canonical)
        decision = decide_from_match_analysis(analysis)
        if analysis.fit_percentage < min_fit:
            continue

        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                st.markdown(f"### {canonical.title}")
                st.write(f"🏢 **{canonical.organization}** · `{canonical.type}`")
                st.markdown("**Requirements before applying:**")
                with st.expander("📋 View full role requirements", expanded=True):
                    st.markdown("**Hard / Mandatory**")
                    for item in canonical.hard_requirements or ["None specified"]:
                        st.write(f"🔴 {item}")
                    st.markdown("**Preferred / Nice to have**")
                    for item in canonical.preferred_requirements or ["None specified"]:
                        st.write(f"🟡 {item}")
            with c2:
                st.metric("Fit", f"{analysis.fit_percentage}%")
                st.metric("Eligibility", analysis.formal_eligibility)
                st.write(f"**Decision:** `{decision['action']}`")
            with c3:
                if st.button("🚀 Apply Now", key=f"apply_{opp_id}", type="primary", use_container_width=True):
                    st.session_state.selected_opp_for_apply = raw_opp
                    st.session_state.current_page = "automation"
                    st.session_state.live_app_id = None
                    st.session_state.live_app_stage = "IDLE"
                    st.session_state.live_app_res = None
                    st.rerun()

            with st.expander("🔍 Decision Breakdown"):
                st.write(f"**Reason:** {decision.get('reason','')}")
                if analysis.evaluations:
                    for ev in analysis.evaluations:
                        marker = "🔴" if ev.status == "FAIL" else "🟡" if ev.status == "UNCERTAIN" else "🟢"
                        st.write(f"{marker} `{ev.requirement.category.value}` · {ev.requirement.raw_text} → **{ev.status}** · {ev.reason}")


# -----------------------------------------------------------------------------
# Automation
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "automation":
    st.markdown("<h2 class='main-header-title'>⚡ Live Automation & Approval</h2>", unsafe_allow_html=True)
    target = st.session_state.selected_opp_for_apply
    if not target:
        st.warning("No opportunity selected. Return to Opportunities and click Apply Now.")
        st.stop()

    st.markdown(f"### 🎯 {target.get('title','Target Role')} — {target.get('organization','Organization')}")
    st.caption("Run #1 → decision → draft/clarification. Human approval → deterministic submission.")

    active_stage = st.session_state.live_app_stage
    st.progress({"IDLE": 0, "CLARIFICATION_REQUIRED": 0.4, "AWAITING_APPROVAL": 0.7, "APPROVED": 0.85, "SUBMITTED": 1.0}.get(active_stage, 0.1))
    st.caption(f"Current stage: `{active_stage}`")

    run_col, reset_col = st.columns([3, 1])
    with run_col:
        if st.button("🚀 Start Autonomous Application", disabled=active_stage not in {"IDLE", "FAILED", "SKIPPED"}, type="primary", use_container_width=True):
            from core.strands_agent import TraceSink
            sink = TraceSink.get_instance()
            live_queue = sink.register_live_queue("*", user_id=user_id)
            result = None
            try:
                with st.status("🤖 ApplyX Agent Running…", expanded=True) as box:
                    def run_workflow():
                        return asyncio.run(orch.process_opportunity(user_id, target, agent_mode=AGENT_MODE))
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_workflow)
                        while not future.done():
                            for event in sink.drain_live_queue(live_queue):
                                if event.get("user_id") != user_id:
                                    continue
                                box.write(f"`[{event.get('run_id')} #{event.get('sequence_number')}]` **{event.get('event_type')}** · `{event.get('tool_name') or ''}`")
                            time.sleep(0.05)
                        for event in sink.drain_live_queue(live_queue):
                            if event.get("user_id") == user_id:
                                box.write(f"`[{event.get('run_id')} #{event.get('sequence_number')}]` **{event.get('event_type')}** · `{event.get('tool_name') or ''}`")
                        result = future.result()
                    box.update(label=f"Workflow complete: {result.stage}", state="complete")
            finally:
                sink.unregister_live_queue("*", live_queue)

            st.session_state.live_app_id = result.application_id
            st.session_state.live_app_stage = result.stage
            st.session_state.live_app_res = result.to_dict()
            st.rerun()

    with reset_col:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.live_app_id = None
            st.session_state.live_app_stage = "IDLE"
            st.session_state.live_app_res = None
            st.session_state.selected_opp_for_apply = None
            st.rerun()

    active_app_id = st.session_state.live_app_id
    result_data = st.session_state.live_app_res or {}

    if active_app_id and active_stage == ApplicationStatus.CLARIFICATION_REQUIRED.value:
        prompt = result_data.get("clarification_prompt") or "Please verify the missing candidate information."
        clarification_id = result_data.get("clarification_id")
        st.warning(f"⚠️ **Clarification required** — {prompt}")
        with st.form("clarification_form"):
            key = st.selectbox("Field", ["cgpa", "graduation_year", "enrollment_status", "work_authorization", "background_check_status", "skills", "years_experience", "projects"])
            value = st.text_input("Verified value")
            send = st.form_submit_button("💬 Submit Clarification & Run #2", type="primary")
            if send:
                if not value.strip():
                    st.error("Value is required.")
                else:
                    parsed: Any = value.strip()
                    if key == "cgpa":
                        try: parsed = float(parsed)
                        except ValueError: pass
                    elif key in {"graduation_year", "years_experience"}:
                        try: parsed = int(parsed)
                        except ValueError: pass
                    elif key == "skills":
                        parsed = [x.strip() for x in parsed.split(",") if x.strip()]
                    with st.status("🤖 Run #2 Re-evaluation…", expanded=True) as box:
                        from core.strands_agent import TraceSink
                        sink = TraceSink.get_instance(); q = sink.register_live_queue(active_app_id, user_id=user_id)
                        try:
                            def run2():
                                return asyncio.run(orch.submit_clarification_and_reevaluate(active_app_id, user_id, target, key, parsed, clarification_id=clarification_id, agent_mode=AGENT_MODE))
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(run2)
                                while not future.done():
                                    for event in sink.drain_live_queue(q):
                                        box.write(f"`[{event.get('run_id')} #{event.get('sequence_number')}]` **{event.get('event_type')}** · `{event.get('tool_name') or ''}`")
                                    time.sleep(0.05)
                                for event in sink.drain_live_queue(q):
                                    box.write(f"`[{event.get('run_id')} #{event.get('sequence_number')}]` **{event.get('event_type')}** · `{event.get('tool_name') or ''}`")
                                res2 = future.result()
                        finally:
                            sink.unregister_live_queue(active_app_id, q)
                        box.update(label=f"Run #2 complete: {res2.stage}", state="complete")
                    st.session_state.live_app_stage = res2.stage
                    st.session_state.live_app_res = res2.to_dict()
                    st.rerun()

    if active_app_id and active_stage in {ApplicationStatus.AWAITING_APPROVAL.value, ApplicationStatus.APPROVED.value}:
        conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        draft_rows = [dict(r) for r in conn.execute("SELECT id,version,fields,answers,documents,created_at FROM application_drafts WHERE application_id=? ORDER BY version", (active_app_id,)).fetchall()]
        app_row = conn.execute("SELECT current_stage,approved_version,approved_draft_id,submitted_version,confirmation_id FROM applications WHERE id=?", (active_app_id,)).fetchone()
        conn.close()

        if draft_rows:
            latest = draft_rows[-1]
            fields = json.loads(latest["fields"])
            answers = json.loads(latest["answers"])
            documents = json.loads(latest["documents"] or "[]")
            st.subheader(f"📄 Draft Workspace — V{latest['version']}")
            st.caption("All prior versions are immutable. Edits create a new version and invalidate prior approval.")

            answer_edits = {}
            for key, payload in answers.items():
                current = payload.get("value", "") if isinstance(payload, dict) else str(payload)
                updated = st.text_area(key.replace("_", " ").title(), value=current, key=f"answer_{active_app_id}_{latest['version']}_{key}")
                answer_edits[key] = {"value": updated.strip(), "source": "user_edited_custom_response", "evidence_ids": payload.get("evidence_ids", []) if isinstance(payload, dict) else []}

            if st.button(f"💾 Save as V{latest['version'] + 1}", type="secondary"):
                revised = orch.create_revised_draft(active_app_id, updated_fields=fields, updated_answers=answer_edits)
                st.success(f"Created immutable Draft V{revised['version']}.")
                st.rerun()

            st.markdown("#### 📎 Resume / Supporting Document")
            uploaded = st.file_uploader("Attach document", type=["pdf", "docx", "txt"], key=f"upload_{active_app_id}_{latest['version']}")
            if uploaded is not None and st.button(f"Attach & Create V{latest['version'] + 1}"):
                upload_dir = ROOT / "data" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                safe_name = os.path.basename(uploaded.name)
                saved = upload_dir / f"{secrets.token_hex(6)}_{safe_name}"
                saved.write_bytes(uploaded.getbuffer())
                revised = orch.create_revised_draft(active_app_id, updated_documents=[{"name": safe_name, "path": str(saved)}])
                st.success(f"Created immutable Draft V{revised['version']} with the uploaded document.")
                st.rerun()

            st.markdown("### 🚦 Human Approval Boundary")
            st.info("The application is intentionally paused here. Open **🛡 Approval Queue** in the sidebar to review the exact agent-filled application before approving it.")
            if st.button("🛡 Open Human Approval Queue", key=f"open_queue_{active_app_id}", use_container_width=True):
                st.session_state.current_page = "approval_queue"
                st.rerun()

            st.markdown("#### Legacy inline approval (same policy gate)")
            versions = [d["version"] for d in draft_rows]
            approved_version = app_row["approved_version"] if app_row else None
            chosen = st.selectbox("Draft version to approve", versions, index=len(versions) - 1)
            if approved_version == chosen and app_row["current_stage"] == ApplicationStatus.APPROVED.value:
                st.success(f"✅ V{chosen} is approved and locked.")
            else:
                if st.button(f"👍 Approve V{chosen}", type="primary"):
                    approval = orch.approve_draft_version(active_app_id, chosen)
                    if approval.get("success"):
                        st.success("Approved.")
                        st.rerun()
                    st.error(approval.get("error", "Approval failed."))

            can_submit = app_row and app_row["current_stage"] == ApplicationStatus.APPROVED.value and app_row["approved_version"] is not None
            if st.button("🌐 Dispatch Approved Version", disabled=not can_submit, type="primary"):
                token = set_orchestrator(orch)
                try:
                    with st.status("🌐 Playwright dispatch…", expanded=True) as box:
                        submission = asyncio.run(submit_application(active_app_id, int(app_row["approved_version"])))
                        if submission.get("success"):
                            box.write(f"✅ Confirmation: `{submission.get('confirmation_id')}`")
                            st.session_state.live_app_stage = ApplicationStatus.SUBMITTED.value
                            st.session_state.live_conf = submission.get("confirmation_id")
                            box.update(label="Submission complete", state="complete")
                        else:
                            box.write(f"🛑 {submission.get('message') or submission.get('error')}")
                            box.update(label="Submission blocked/failed", state="error")
                finally:
                    reset_orchestrator(token)
                st.rerun()

    if active_app_id and active_stage == ApplicationStatus.SKIPPED.value:
        st.error(f"🛑 SKIPPED — {result_data.get('reason','Hard criterion failed.')}")
    if active_app_id and active_stage == ApplicationStatus.FAILED.value:
        st.error(f"❌ FAILED — {result_data.get('reason','Workflow failure.')}")
    if active_app_id and active_stage == ApplicationStatus.SUBMITTED.value:
        st.success(f"🎉 SUBMITTED — `{st.session_state.get('live_conf','')}`")

    if active_app_id:
        conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        traces = [dict(r) for r in conn.execute("SELECT run_id,sequence_number,event_type,stage,tool_name,event_payload,created_at FROM application_events WHERE application_id=? ORDER BY id", (active_app_id,)).fetchall()]
        conn.close()
        with st.expander(f"🔍 Authoritative Trace ({len(traces)} events)"):
            for trace in traces:
                st.write(f"`[{trace['run_id']} #{trace['sequence_number']}]` **{trace['event_type']}** · `{trace['tool_name'] or ''}` · {trace['created_at'][:19]}")


# -----------------------------------------------------------------------------
# Human Approval Queue
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "approval_queue":
    _render_approval_queue()


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "history":
    st.markdown("<h2 class='main-header-title'>📜 Application History</h2>", unsafe_allow_html=True)
    conn = sqlite3.connect(DB_FILE_PATH, timeout=30.0)
    rows = conn.execute(
        """
        SELECT a.id,a.opportunity_id,COALESCE(o.title,a.opportunity_id),COALESCE(o.organization,'Direct Employer'),COALESCE(o.type,'Role'),a.current_stage,a.approved_version,a.submitted_version,a.confirmation_id,a.updated_at
        FROM applications a LEFT JOIN opportunities o ON a.opportunity_id=o.id WHERE a.user_id=? ORDER BY a.rowid DESC
        """, (user_id,)
    ).fetchall()
    conn.close()
    if not rows:
        st.info("No application records yet.")
    else:
        df = pd.DataFrame(rows, columns=["Application ID","Opportunity ID","Opportunity","Organization","Type","Stage","Approved Version","Submitted Version","Confirmation","Updated"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Export CSV", df.to_csv(index=False), file_name=f"applyx_history_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "benchmark":
    st.markdown("<h2 class='main-header-title'>📊 Reliability Benchmark</h2>", unsafe_allow_html=True)
    a_tab, b_tab = st.tabs(["🎯 Mode A — Deterministic", "🤖 Mode B — Strands Safety Harness"])

    with a_tab:
        if st.button("▶️ Run Mode A", type="primary"):
            st.session_state.bench_a = run_deterministic_benchmark()
            st.rerun()
        if st.session_state.get("bench_a"):
            result = st.session_state.bench_a
            c1, c2, c3 = st.columns(3)
            c1.metric("Action Accuracy", f"{result['action_accuracy']:.1f}%")
            c2.metric("Eligibility Accuracy", f"{result['eligibility_accuracy']:.1f}%")
            c3.metric("Hard Leaks", result["hard_leaks"])
            st.dataframe(pd.DataFrame(result["results"]), use_container_width=True, hide_index=True)

    with b_tab:
        mode = st.selectbox("Agent backend", ["mock", "real"], index=0, key="bench_agent_mode")
        if mode == "real" and not has_real_credentials():
            st.warning("REAL Mode B is unavailable until model credentials are configured. MOCK is explicit and independent.")
        if st.button("▶️ Run Mode B", type="primary", disabled=(mode == "real" and not has_real_credentials())):
            st.session_state.bench_b = run_agent_workflow_benchmark(mode)
            st.rerun()
        if st.session_state.get("bench_b"):
            result = st.session_state.bench_b
            c1, c2, c3 = st.columns(3)
            c1.metric("Reliability", f"{result['accuracy']:.1f}%")
            c2.metric("Passed", f"{result['passed']}/{result['total']}")
            c3.metric("Mode", result["mode"].upper())
            st.dataframe(pd.DataFrame(result["results"]), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Profile
# -----------------------------------------------------------------------------
elif st.session_state.current_page == "profile":
    st.markdown("<h2 class='main-header-title'>⚙️ Candidate Profile & Knowledge Base</h2>", unsafe_allow_html=True)
    st.write(f"Mandatory: **{completeness['mandatory']}%** · Optional: **{completeness['optional']}%** · Total: **{completeness['total']}%**")

    with st.form("profile_form"):
        st.markdown("### 👤 Personal & Contact")
        c1, c2, c3 = st.columns(3)
        with c1:
            full_name = st.text_input("Full Name", value=profile.get("full_name") or "")
            email = st.text_input("Email", value=profile.get("email") or "")
            phone = st.text_input("Phone", value=profile.get("phone") or "")
        with c2:
            location = st.text_input("Location", value=profile.get("location") or "")
            linkedin = st.text_input("LinkedIn URL", value=profile.get("linkedin_url") or "")
            github = st.text_input("GitHub URL", value=profile.get("github_url") or "")
        with c3:
            portfolio = st.text_input("Portfolio URL", value=profile.get("portfolio_url") or "")
            current_photo = profile.get("profile_photo_path") or ""
            st.caption(f"Current photo: `{current_photo}`" if current_photo else "No profile photo uploaded")
            current_resume = profile.get("resume_path") or ""
            st.caption(f"Current resume: `{profile.get('resume_name') or os.path.basename(current_resume)}`" if current_resume else "No resume uploaded")

        st.markdown("### 🎓 Education")
        e1, e2, e3 = st.columns(3)
        with e1:
            degree = st.text_input("Current / Highest Degree", value=profile.get("degree") or "")
        with e2:
            college_name = st.text_input("College / University", value=profile.get("college_name") or "")
        with e3:
            grad = st.number_input("Graduation Year (0 = unspecified)", min_value=0, max_value=2035, value=int(profile.get("graduation_year") or 0))
        education_text = st.text_area(
            "Education History (one entry per line: Degree | College | Year)",
            value=_format_education(profile.get("education_history")),
            height=100,
        )

        st.markdown("### 💼 Work Experience")
        st.caption("Add company names and roles rather than only a total number of years.")
        work_text = st.text_area(
            "Work Experience (one per line: Role | Company | Period)",
            value=_format_experience(profile.get("work_experience")),
            height=120,
        )
        years = st.number_input(
            "Total Years of Experience (optional, used only when a role explicitly asks for years)",
            min_value=0,
            max_value=30,
            value=int(profile.get("years_experience") or 0),
        )

        st.markdown("### 🧠 Skills, Projects & Eligibility")
        c1, c2 = st.columns(2)
        with c1:
            skills = st.text_area("Skills (comma-separated)", value=", ".join(profile.get("skills") or []))
            projects = st.text_area("Projects (one per line)", value="\n".join(profile.get("projects") or []), height=120)
        with c2:
            work_options = ["Unknown", "Yes", "No"]
            current_wa = profile.get("work_authorization") or "Unknown"
            work_auth = st.selectbox("Work Authorization", work_options, index=work_options.index(current_wa) if current_wa in work_options else 0)
            enrollment_options = ["UNKNOWN", "CURRENTLY_ENROLLED", "GRADUATED", "NOT_CURRENTLY_ENROLLED"]
            current_enr = profile.get("enrollment_status") or "UNKNOWN"
            enrollment = st.selectbox("Enrollment Status", enrollment_options, index=enrollment_options.index(current_enr) if current_enr in enrollment_options else 0)
            bg_options = ["UNKNOWN", "PASS", "FAIL"]
            current_bg = profile.get("background_check_status") or "UNKNOWN"
            background = st.selectbox("Background Check", bg_options, index=bg_options.index(current_bg) if current_bg in bg_options else 0)
            cgpa_value = st.number_input("CGPA (0 = unspecified)", min_value=0.0, max_value=10.0, value=float(profile["cgpa"]) if profile.get("cgpa") is not None else 0.0, step=0.01)

        save = st.form_submit_button("💾 Save Ground-Truth Profile", type="primary", use_container_width=True)

    st.markdown("### 📎 Upload Resume & Profile Photo")
    resume_upload = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key=f"resume_profile_{user_id}")
    photo_upload = st.file_uploader("Profile Photo (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"], key=f"photo_profile_{user_id}")
    existing_photo = profile.get("profile_photo_path")
    if existing_photo and os.path.exists(existing_photo):
        st.image(existing_photo, caption="Current profile photo", width=150)

    if save:
        def _parse_pipe_rows(text: str, keys: list[str]) -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            for line in text.splitlines():
                parts = [part.strip() for part in line.split("|")]
                if not any(parts):
                    continue
                row = {key: (parts[i] if i < len(parts) else "") for i, key in enumerate(keys)}
                if any(row.values()):
                    rows.append(row)
            return rows

        values = {
            "full_name": full_name.strip() or None,
            "email": email.strip() or None,
            "phone": phone.strip() or None,
            "location": location.strip() or None,
            "linkedin_url": linkedin.strip() or None,
            "github_url": github.strip() or None,
            "portfolio_url": portfolio.strip() or None,
            "degree": degree.strip() or None,
            "college_name": college_name.strip() or None,
            "education_history": _parse_pipe_rows(education_text, ["degree", "college", "graduation_year"]),
            "graduation_year": int(grad) if grad > 0 else None,
            "work_experience": _parse_pipe_rows(work_text, ["role", "company", "period"]),
            "years_experience": int(years),
            "skills": [x.strip() for x in skills.split(",") if x.strip()],
            "projects": [x.strip() for x in projects.splitlines() if x.strip()],
            "work_authorization": None if work_auth == "Unknown" else work_auth,
            "enrollment_status": None if enrollment == "UNKNOWN" else enrollment,
            "background_check_status": None if background == "UNKNOWN" else background,
            "cgpa": cgpa_value if cgpa_value > 0 else None,
        }
        for field, value in values.items():
            orch.update_user_kb(user_id, field, value)

        upload_dir = ROOT / "data" / "uploads" / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        if resume_upload is not None:
            safe_name = os.path.basename(resume_upload.name)
            saved = upload_dir / f"resume_{secrets.token_hex(6)}_{safe_name}"
            saved.write_bytes(resume_upload.getbuffer())
            orch.update_user_kb(user_id, "resume_path", str(saved))
            orch.update_user_kb(user_id, "resume_name", safe_name)
        if photo_upload is not None:
            safe_name = os.path.basename(photo_upload.name)
            saved = upload_dir / f"photo_{secrets.token_hex(6)}_{safe_name}"
            saved.write_bytes(photo_upload.getbuffer())
            orch.update_user_kb(user_id, "profile_photo_path", str(saved))

        st.success("✅ Profile, education, work history and links saved to the Knowledge Base.")
        st.rerun()

    st.markdown("### 🧩 Ground-Truth KB")
    st.json(orch.fetch_user_profile(user_id))
