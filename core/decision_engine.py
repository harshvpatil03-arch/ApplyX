"""
core/decision_engine.py
Policy authority for ApplyX.

Priority:
1. Blocking hard failure -> SKIP
2. Blocking hard uncertainty -> REVIEW
3. Conditional ambiguity -> REVIEW
4. Preferred/soft gaps -> APPLY

The decision is derived from structured evaluation evidence, not from model prose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from core.schemas import MatchAnalysisResult, SkillRelation


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iter_evaluations(analysis: Any) -> Iterable[Any]:
    return _field(analysis, "evaluations", []) or []


def decide_from_match_analysis(analysis: MatchAnalysisResult | Dict[str, Any]) -> Dict[str, Any]:
    """Return the deterministic APPLY/REVIEW/SKIP policy decision."""
    confidence = float(_field(analysis, "reasoning_confidence", 0.85) or 0.85)
    confidence_label = str(_field(analysis, "reasoning_confidence_label", "HIGH"))

    missing_hard = list(_field(analysis, "missing_hard", []) or [])
    uncertain_criteria = list(_field(analysis, "uncertain_criteria", []) or [])
    conditional_criteria = list(_field(analysis, "conditional_criteria", []) or [])
    missing_pref = list(_field(analysis, "missing_pref", []) or [])
    fit_score = int(_field(analysis, "fit_percentage", 0) or 0)

    why_apply = []
    why_not_skip = []
    why_review = []

    for ev in _iter_evaluations(analysis):
        status = str(_field(ev, "status", "")).upper()
        blocking = bool(_field(ev, "blocking", False))
        relation = _field(ev, "relation", SkillRelation.UNKNOWN)
        raw_text = str(_field(_field(ev, "requirement", {}), "raw_text", "Requirement"))
        reason = str(_field(ev, "reason", ""))
        evidence = _field(ev, "evidence", []) or []

        relation_value = relation.value if isinstance(relation, SkillRelation) else str(relation)
        if status == "PASS":
            evidence_source = "Verified in profile"
            if evidence:
                evidence_source = str(_field(evidence[0], "source", evidence_source))
            why_apply.append({
                "text": raw_text,
                "relation": "Direct match" if relation_value == SkillRelation.DIRECT.value else f"{relation_value} capability match",
                "evidence": f"Evidence: {evidence_source}",
            })
        elif status == "UNCERTAIN" and blocking:
            why_review.append({"text": raw_text, "reason": reason})
        elif status == "UNCERTAIN" and not blocking:
            why_not_skip.append({"text": raw_text, "reason": f"Non-blocking administrative uncertainty: {reason}"})
        elif status == "FAIL" and not blocking:
            why_not_skip.append({"text": raw_text, "reason": f"Soft deficit: {reason}"})

    has_hard_failure = any(
        str(_field(ev, "status", "")).upper() == "FAIL" and bool(_field(ev, "blocking", False))
        for ev in _iter_evaluations(analysis)
    ) or bool(missing_hard)

    has_blocking_uncertainty = any(
        str(_field(ev, "status", "")).upper() == "UNCERTAIN" and bool(_field(ev, "blocking", False))
        for ev in _iter_evaluations(analysis)
    ) or bool(uncertain_criteria)

    if has_hard_failure:
        failed = ", ".join(missing_hard) or "blocking hard requirement"
        return {
            "action": "SKIP",
            "formal_eligibility": "NO",
            "confidence": confidence,
            "confidence_label": confidence_label,
            "fit_score": fit_score,
            "risk": "CRITICAL",
            "reason": f"Disqualified: hard requirement failed ({failed}).",
            "clarification_prompt": None,
            "why_apply": why_apply,
            "why_not_skip": [],
            "why_review": [],
        }

    if has_blocking_uncertainty:
        criterion = uncertain_criteria[0] if uncertain_criteria else "mandatory profile attribute"
        return {
            "action": "REVIEW",
            "formal_eligibility": "UNCERTAIN",
            "confidence": confidence,
            "confidence_label": confidence_label,
            "fit_score": fit_score,
            "risk": "MEDIUM",
            "reason": f"Information required: candidate evidence is missing for '{criterion}'.",
            "clarification_prompt": f"Please verify your status for: '{criterion}'.",
            "why_apply": why_apply,
            "why_not_skip": why_not_skip,
            "why_review": why_review or [{"text": criterion, "reason": "Blocking evidence is missing."}],
        }

    if conditional_criteria:
        cond = conditional_criteria[0]
        return {
            "action": "REVIEW",
            "formal_eligibility": "YES",
            "confidence": confidence,
            "confidence_label": confidence_label,
            "fit_score": fit_score,
            "risk": "MEDIUM",
            "reason": f"Conditional criterion requires human review: {cond}",
            "clarification_prompt": None,
            "why_apply": why_apply,
            "why_not_skip": why_not_skip,
            "why_review": [{"text": cond, "reason": "Subject to external or managerial discretion."}],
        }

    if missing_hard or not uncertain_criteria:
        why_not_skip.insert(0, {
            "text": "No blocking hard criterion failed",
            "reason": "All blocking hard requirements are satisfied or remain non-blocking administrative items.",
        })

    return {
        "action": "APPLY",
        "formal_eligibility": "YES",
        "confidence": confidence,
        "confidence_label": confidence_label,
        "fit_score": fit_score,
        "risk": "LOW",
        "reason": (
            "All mandatory criteria satisfied; preferred gaps are non-blocking."
            if missing_pref
            else "All mandatory and preferred criteria satisfied."
        ),
        "clarification_prompt": None,
        "why_apply": why_apply,
        "why_not_skip": why_not_skip,
        "why_review": [],
    }
