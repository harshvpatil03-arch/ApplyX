"""
core/agent_tools.py
Thin, controlled Strands tools for ApplyX.

The tools deliberately trust invocation context for infrastructure state:
application_id, user_id, opportunity payload, clarification id, and attempt.
LLM-supplied infrastructure IDs are not authoritative.
"""

from __future__ import annotations

from dataclasses import asdict
import contextvars
import logging
import re
import sqlite3
from typing import Any, Dict, Optional
import uuid

try:
    from strands import tool
except ImportError:
    # Only permits importing the module in environments where the real SDK is not installed.
    # Real agent execution is rejected by core.strands_agent rather than silently simulated here.
    def tool(fn=None, **kwargs):
        if fn is not None and callable(fn):
            return fn
        def decorator(function):
            return function
        return decorator

from core.decision_engine import decide_from_match_analysis
from core.matching_engine import evaluate_candidate_match
from core.schemas import CanonicalOpportunity, MatchAnalysisResult

logger = logging.getLogger(__name__)

_orchestrator_var: contextvars.ContextVar = contextvars.ContextVar("applyx_orchestrator", default=None)


def set_orchestrator(orchestrator_inst):
    """Bind an orchestrator to the current async/contextvar scope and return a reset token."""
    return _orchestrator_var.set(orchestrator_inst)


def reset_orchestrator(token) -> None:
    _orchestrator_var.reset(token)


def get_orchestrator():
    orchestrator = _orchestrator_var.get()
    if orchestrator is None:
        raise RuntimeError("No orchestrator bound to current agent invocation context.")
    return orchestrator


def _context():
    from core.strands_agent import _context_invocation
    return _context_invocation.get()


def _trusted_user_id(requested: Optional[str] = None) -> str:
    ctx = _context()
    if ctx:
        if requested and requested != ctx.user_id:
            raise ValueError("LLM-supplied user_id does not match trusted invocation context.")
        return ctx.user_id
    if requested:
        return requested
    raise RuntimeError("Trusted user_id is unavailable outside an invocation context.")


def _trusted_opportunity(requested: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = _context()
    if ctx and getattr(ctx, "opportunity_payload", None) is not None:
        return dict(ctx.opportunity_payload)
    if isinstance(requested, dict):
        return dict(requested)
    raise RuntimeError("Trusted opportunity payload is unavailable.")


def _analysis_dict(analysis: MatchAnalysisResult) -> Dict[str, Any]:
    return analysis.to_dict()


@tool
def get_candidate_profile(user_id: str) -> Dict[str, Any]:
    """Read the trusted candidate Knowledge Base profile."""
    orch = get_orchestrator()
    trusted_id = _trusted_user_id(user_id)
    return orch.fetch_user_profile(trusted_id)


@tool
def normalize_opportunity(opportunity: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the trusted opportunity into CanonicalOpportunity."""
    orch = get_orchestrator()
    canonical = orch.normalize_opportunity(_trusted_opportunity(opportunity))
    return canonical.to_dict()


@tool
def analyze_candidate_match(user_profile: Dict[str, Any], opportunity: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate candidate eligibility from the trusted KB and trusted opportunity."""
    orch = get_orchestrator()
    trusted_user = _trusted_user_id()
    trusted_profile = orch.fetch_user_profile(trusted_user)
    canonical = orch.normalize_opportunity(_trusted_opportunity(opportunity))
    analysis = evaluate_candidate_match(trusted_profile, canonical)
    return _analysis_dict(analysis)


@tool
def make_application_decision(match_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the previously captured deterministic analysis into APPLY/REVIEW/SKIP."""
    ctx = _context()
    source_analysis = None
    if ctx:
        source_analysis = ctx.tool_results.get("analyze_candidate_match") or ctx.tool_results.get("re_evaluate_application", {}).get("match_analysis")
    if ctx and source_analysis is None:
        raise RuntimeError("No deterministic match analysis is available for the decision tool.")
    source_analysis = source_analysis or match_analysis
    decision = decide_from_match_analysis(source_analysis)
    if ctx:
        ctx.decision_result = decision
    return decision


@tool
def generate_application_draft(
    user_id: str,
    opportunity: Dict[str, Any],
    match_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate immutable Draft V1 using only trusted invocation state."""
    orch = get_orchestrator()
    ctx = _context()
    if not ctx:
        return {"status": "ERROR", "error_type": "MISSING_INVOCATION_CONTEXT", "message": "Trusted application context is required."}
    if ctx.decision_result and ctx.decision_result.get("action") != "APPLY":
        return {"status": "ERROR", "error_type": "INVALID_BRANCH", "message": "Draft generation is only valid after an authoritative APPLY decision."}
    trusted_user = _trusted_user_id(user_id)
    canonical = orch.normalize_opportunity(_trusted_opportunity(opportunity))
    return orch.generate_and_persist_draft(ctx.application_id, orch.fetch_user_profile(trusted_user), canonical)


@tool
def request_clarification(user_id: str, field: str, reason: str) -> Dict[str, Any]:
    """Create one durable clarification request context for the current application."""
    ctx = _context()
    trusted_id = _trusted_user_id(user_id)
    if not ctx:
        raise RuntimeError("Clarification requires invocation context.")
    if not field or field not in {
        "full_name", "degree", "cgpa", "graduation_year",
        "enrollment_status", "work_authorization",
        "background_check_status", "years_experience", "skills", "projects"
    }:
        raise ValueError(f"Unsupported clarification field: {field}")

    if not ctx.clarification_id:
        ctx.clarification_id = f"CLR-{uuid.uuid4()}"
    if not re.fullmatch(r"CLR-[0-9a-fA-F-]{36}", ctx.clarification_id):
        raise ValueError("Invalid clarification identifier.")

    payload = {
        "status": "CLARIFICATION_REQUIRED",
        "clarification_id": ctx.clarification_id,
        "application_id": ctx.application_id,
        "user_id": trusted_id,
        "field": field,
        "reason": reason,
        "prompt": f"ApplyX requires verified '{field}' information to continue eligibility evaluation ({reason}).",
    }
    orch = get_orchestrator()
    try:
        orch.log_audit_event(
            ctx.application_id,
            "CLARIFICATION_REQUESTED",
            payload,
            run_id=ctx.run_id,
            event_type="CLARIFICATION_REQUESTED",
        )
    except Exception:
        logger.exception("Failed to persist clarification request")
        raise
    return payload


@tool
def re_evaluate_application(
    user_id: str,
    opportunity: Dict[str, Any],
    clarification_key: str,
    clarification_answer: Any,
    current_attempts: int = 0,
) -> Dict[str, Any]:
    """Re-evaluate only; KB mutation happens once in the orchestrator before Run #2."""
    del clarification_key, clarification_answer, current_attempts
    orch = get_orchestrator()
    ctx = _context()
    if not ctx:
        raise RuntimeError("Re-evaluation requires invocation context.")
    trusted_user = _trusted_user_id(user_id)
    trusted_profile = orch.fetch_user_profile(trusted_user)
    canonical = orch.normalize_opportunity(_trusted_opportunity(opportunity))
    analysis = evaluate_candidate_match(trusted_profile, canonical)
    analysis_dict = _analysis_dict(analysis)
    return {
        "user_profile": trusted_profile,
        "match_analysis": analysis_dict,
        "decision": decide_from_match_analysis(analysis_dict),
        "attempts": ctx.clarification_attempt,
    }


@tool
async def submit_application(application_id: str, version: int) -> Dict[str, Any]:
    """Submission entry point. Not included in either Strands intake toolset."""
    orch = get_orchestrator()
    if not isinstance(application_id, str) or not application_id.strip():
        return {"success": False, "error_type": "INVALID_APPLICATION_ID", "message": "Application ID is required."}
    try:
        version = int(version)
    except (TypeError, ValueError):
        return {"success": False, "error_type": "INVALID_VERSION", "message": "Draft version must be an integer."}
    if version < 1:
        return {"success": False, "error_type": "INVALID_VERSION", "message": "Draft version must be >= 1."}

    conn = orch._get_db_connection()
    try:
        row = conn.execute(
            "SELECT approved_version, submitted_version, confirmation_id FROM applications WHERE id = ?",
            (application_id,),
        ).fetchone()
    finally:
        conn.close()

    # Required ordering: idempotent replay first, then security gate, then browser.
    if row and row[0] == version and row[1] == version and row[2]:
        return {
            "success": True,
            "status": "ALREADY_SUBMITTED",
            "application_id": application_id,
            "submitted_version": version,
            "confirmation_id": row[2],
        }

    safe, errors = orch.pre_submission_policy_gate(application_id, version)
    if not safe:
        return {
            "success": False,
            "error_type": "POLICY_GATE_REJECTION",
            "errors": errors,
            "message": f"Policy Gate Blocked Submission: {' | '.join(errors)}",
        }

    try:
        return await orch._execute_browser_submission(application_id, version)
    except Exception as exc:
        logger.exception("Browser submission execution exception")
        return {"success": False, "error_type": "EXECUTION_EXCEPTION", "error": str(exc)}
