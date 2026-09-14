"""
diagnostics/run_benchmark.py
Authoritative ApplyX evaluation harness.

Mode A: deterministic matching/decision benchmark (30 fixture cases, no LLM/network/browser).
Mode B: five Strands workflow safety scenarios, explicitly REAL or MOCK.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from core.agent_tools import set_orchestrator, reset_orchestrator, submit_application
from core.orchestrator import ApplicationOrchestrator
from core.matching_engine import evaluate_candidate_match
from core.decision_engine import decide_from_match_analysis

FIXTURE_PATH = ROOT / "data" / "evaluation" / "test_cases.json"


def _load_cases() -> List[Dict[str, Any]]:
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(f"Benchmark fixture not found: {FIXTURE_PATH}")

    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0.0"
        or not isinstance(payload.get("cases"), list)
    ):
        raise ValueError(
            "Benchmark fixture must contain schema_version=1.0.0 and a cases list."
        )

    cases = payload["cases"]
    validate_test_fixture(cases)
    return cases


def validate_test_fixture(cases: List[Dict[str, Any]]) -> None:
    if len(cases) != 30:
        raise ValueError(
            f"Expected exactly 30 canonical benchmark cases, found {len(cases)}"
        )

    seen = set()

    valid_actions = {
        "APPLY",
        "REVIEW",
        "SKIP",
        "DUPLICATE",
    }

    valid_eligibility = {
        "YES",
        "NO",
        "UNCERTAIN",
        "N/A",
    }

    for case in cases:
        cid = case.get("id")

        if not cid or cid in seen:
            raise ValueError(
                f"Invalid or duplicate case id: {cid!r}"
            )

        seen.add(cid)

        if case.get("case_id") != cid:
            raise ValueError(
                f"Case '{cid}' must have matching case_id."
            )

        if (
            not isinstance(case.get("user_profile"), dict)
            or not isinstance(case.get("opportunity"), dict)
        ):
            raise ValueError(
                f"Case '{cid}' requires user_profile and opportunity objects."
            )

        if case.get("expected_action") not in valid_actions:
            raise ValueError(
                f"Case '{cid}' has invalid expected_action."
            )

        if case.get("expected_formal_eligibility") not in valid_eligibility:
            raise ValueError(
                f"Case '{cid}' has invalid expected_formal_eligibility."
            )


def _fresh_db(prefix: str) -> str:
    path = os.path.join(
        tempfile.gettempdir(),
        f"{prefix}_{uuid.uuid4().hex}.db",
    )
    return path


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def run_deterministic_benchmark() -> Dict[str, Any]:
    cases = _load_cases()

    action_matches = 0
    eligibility_matches = 0
    hard_leaks = 0
    false_apply_ambiguous = 0
    blocking_uncertainty_errors = 0

    results: List[Dict[str, Any]] = []

    for case in cases:
        db_path = _fresh_db("applyx_mode_a")

        kb_dir = os.path.join(
            tempfile.gettempdir(),
            f"applyx_mode_a_kb_{uuid.uuid4().hex}",
        )

        orch = ApplicationOrchestrator(
            db_path=db_path,
            kb_dir=kb_dir,
        )

        try:
            profile = case["user_profile"]

            opportunity = orch.normalize_opportunity(
                case["opportunity"]
            )

            analysis = evaluate_candidate_match(
                profile,
                opportunity,
            )

            decision = decide_from_match_analysis(
                analysis
            )

            expected_action = case["expected_action"]
            expected_eligibility = case[
                "expected_formal_eligibility"
            ]
            expected_risk = case.get("expected_risk")

            # Duplicate cases use the real authoritative
            # duplicate lookup against a seeded record.
            if expected_action == "DUPLICATE":
                conn = orch._get_db_connection()

                try:
                    now = "2026-01-01T00:00:00"

                    conn.execute(
                        """
                        INSERT INTO applications
                        (
                            id,
                            user_id,
                            opportunity_id,
                            current_stage,
                            status,
                            created_at,
                            updated_at
                        )
                        VALUES
                        (
                            ?,
                            ?,
                            ?,
                            'SUBMITTED',
                            'SUBMITTED',
                            ?,
                            ?
                        )
                        """,
                        (
                            f"EXISTING-{case['id']}",
                            profile.get("id", "USER-EVAL"),
                            opportunity.id,
                            now,
                            now,
                        ),
                    )

                    conn.commit()

                finally:
                    conn.close()

                actual_action = (
                    "DUPLICATE"
                    if orch.check_duplicate(
                        profile.get("id", "USER-EVAL"),
                        opportunity.id,
                    )
                    else decision["action"]
                )

            else:
                actual_action = decision["action"]

            actual_eligibility = analysis.formal_eligibility

            actual_risk = (
                "NONE"
                if actual_action == "DUPLICATE"
                else decision.get("risk", "UNKNOWN")
            )

            action_ok = (
                actual_action == expected_action
            )

            elig_ok = (
                actual_eligibility == expected_eligibility
            )

            risk_ok = (
                expected_risk is None
                or actual_risk == expected_risk
            )

            action_matches += int(action_ok)
            eligibility_matches += int(elig_ok)

            if (
                expected_action == "SKIP"
                and actual_action == "APPLY"
            ):
                hard_leaks += 1

            if (
                expected_action == "REVIEW"
                and actual_action == "APPLY"
                and case.get("category") == "MISSING_INFO"
            ):
                false_apply_ambiguous += 1

            if (
                expected_eligibility == "UNCERTAIN"
                and not elig_ok
            ):
                blocking_uncertainty_errors += 1

            results.append(
                {
                    "id": case["id"],
                    "name": case.get(
                        "name",
                        case["id"],
                    ),
                    "expected_action": expected_action,
                    "actual_action": actual_action,
                    "expected_eligibility": expected_eligibility,
                    "actual_eligibility": actual_eligibility,
                    "expected_risk": expected_risk,
                    "actual_risk": actual_risk,
                    "risk_ok": risk_ok,
                    "status": (
                        "PASS"
                        if action_ok
                        and elig_ok
                        and risk_ok
                        else "FAIL"
                    ),
                    "reason": decision.get(
                        "reason",
                        "",
                    ),
                }
            )

        finally:
            _remove(db_path)

    total = len(cases)

    return {
        "mode": "deterministic",
        "status": (
            "PASS"
            if action_matches == total
            and eligibility_matches == total
            and hard_leaks == 0
            and all(
                r["risk_ok"]
                for r in results
            )
            else "FAIL"
        ),
        "action_accuracy": (
            action_matches / total * 100.0
        ),
        "eligibility_accuracy": (
            eligibility_matches / total * 100.0
        ),
        "hard_leaks": hard_leaks,
        "false_apply_ambiguous": false_apply_ambiguous,
        "blocking_uncertainty_errors": blocking_uncertainty_errors,
        "total": total,
        "passed": action_matches,
        "results": results,
    }


def _browser_call_counter(
    orch: ApplicationOrchestrator,
):
    counter = {
        "calls": 0
    }

    original = orch._execute_browser_submission

    async def counted(*args, **kwargs):
        counter["calls"] += 1
        return await original(
            *args,
            **kwargs,
        )

    orch._execute_browser_submission = counted  # type: ignore[method-assign]

    return counter


def _run_scenario(
    orch: ApplicationOrchestrator,
    user: Dict[str, Any],
    opp: Dict[str, Any],
    mode: str,
) -> Any:

    orch.fetch_user_profile = (
        lambda uid, p=user: p
    )

    return asyncio.run(
        orch.process_opportunity(
            user["id"],
            opp,
            agent_mode=mode,
        )
    )


def run_agent_workflow_benchmark(
    agent_mode: str = "mock",
) -> Dict[str, Any]:

    mode = agent_mode.strip().lower()

    if mode not in {
        "mock",
        "real",
    }:
        raise ValueError(
            "agent_mode must be 'mock' or 'real'."
        )

    import core.orchestrator as orchestrator_module

    previous_demo = orchestrator_module.DEMO_MODE

    orchestrator_module.DEMO_MODE = (
        mode == "mock"
    )

    results: List[Dict[str, Any]] = []

    db_paths: List[str] = []
    kb_dirs: List[str] = []

    try:

        # ============================================================
        # AG-01
        # ============================================================

        db = _fresh_db("applyx_ag01")
        db_paths.append(db)

        kb = db + ".kb"
        kb_dirs.append(kb)

        orch = ApplicationOrchestrator(
            db_path=db,
            kb_dir=kb,
        )

        user = {
            "id": "USER-AG-01",
            "full_name": "Alice Candidate",
            "degree": "B.Tech Computer Science",
            "cgpa": 8.8,
            "skills": [
                "Python",
                "Playwright",
                "SQLite",
            ],
            "years_experience": 2,
        }

        opp = {
            "id": "OPP-AG-01",
            "title": "Software Engineer Intern",
            "organization": "Alpha AI",
            "type": "Internship",
            "hard_requirements": [
                "Python",
                "Degree in CS or related",
            ],
        }

        res = _run_scenario(
            orch,
            user,
            opp,
            mode,
        )

        # IMPORTANT:
        # Use one explicit SQLite connection for both queries
        # and always close it.
        #
        # This fixes the Windows WinError 32 caused by the
        # previous leaked connections:
        #
        # orch._get_db_connection().execute(...)
        #
        # Those chained connections were never closed.

        conn = orch._get_db_connection()

        try:
            row = conn.execute(
                """
                SELECT
                    current_stage,
                    approved_version
                FROM applications
                WHERE id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()

            drafts = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_drafts
                WHERE application_id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

        finally:
            conn.close()

        conn = orch._get_db_connection()

        try:
            trace_rows = conn.execute(
                """
                SELECT
                    run_id,
                    sequence_number,
                    tool_name,
                    event_type
                FROM application_events
                WHERE application_id = ?
                  AND run_id = 'RUN-1'
                ORDER BY sequence_number
                """,
                (
                    res.application_id,
                ),
            ).fetchall()

        finally:
            conn.close()

        tool_names = [
            r[2]
            for r in trace_rows
            if r[3] == "STRANDS_BEFORE_TOOL_CALL"
        ]

        expected = [
            "get_candidate_profile",
            "normalize_opportunity",
            "analyze_candidate_match",
            "make_application_decision",
            "generate_application_draft",
        ]

        ok = (
            res.decision == "APPLY"
            and res.stage == "AWAITING_APPROVAL"
            and row
            and row[0] == "AWAITING_APPROVAL"
            and row[1] is None
            and drafts == 1
            and tool_names == expected
        )

        results.append(
            {
                "id": "AG-01",
                "name": "Clear APPLY -> Draft V1",
                "passed": bool(ok),
                "details": (
                    f"mode={mode}, "
                    f"tools={tool_names}, "
                    f"drafts={drafts}"
                ),
            }
        )

        # ============================================================
        # AG-02
        # ============================================================

        db = _fresh_db("applyx_ag02")
        db_paths.append(db)

        kb = db + ".kb"
        kb_dirs.append(kb)

        orch = ApplicationOrchestrator(
            db_path=db,
            kb_dir=kb,
        )

        user = {
            "id": "USER-AG-02",
            "full_name": "Bob Student",
            "degree": "B.Tech AI",
            "skills": [
                "Python"
            ],
            "years_experience": 1,
        }

        opp = {
            "id": "OPP-AG-02",
            "title": "Graduate Trainee",
            "organization": "Beta Corp",
            "type": "Full-Time",
            "hard_requirements": [
                "Python",
                "Minimum CGPA 8.0 cut-off",
            ],
        }

        res1 = _run_scenario(
            orch,
            user,
            opp,
            mode,
        )

        clr_id = res1.clarification_id

        res2 = asyncio.run(
            orch.submit_clarification_and_reevaluate(
                res1.application_id,
                user["id"],
                opp,
                "cgpa",
                8.5,
                clarification_id=clr_id,
                agent_mode=mode,
            )
        )

        conn = orch._get_db_connection()

        try:
            app_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM applications
                WHERE user_id = ?
                  AND opportunity_id = ?
                """,
                (
                    user["id"],
                    opp["id"],
                ),
            ).fetchone()[0]

            kb_events = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_events
                WHERE application_id = ?
                  AND event_type = 'USER_CLARIFICATION'
                """,
                (
                    res1.application_id,
                ),
            ).fetchone()[0]

            draft_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_drafts
                WHERE application_id = ?
                """,
                (
                    res1.application_id,
                ),
            ).fetchone()[0]

            r2_tools = [
                r[0]
                for r in conn.execute(
                    """
                    SELECT tool_name
                    FROM application_events
                    WHERE application_id = ?
                      AND run_id = 'RUN-2'
                      AND event_type = 'STRANDS_BEFORE_TOOL_CALL'
                    ORDER BY sequence_number
                    """,
                    (
                        res1.application_id,
                    ),
                ).fetchall()
            ]

        finally:
            conn.close()

        expected_r2 = [
            "get_candidate_profile",
            "normalize_opportunity",
            "re_evaluate_application",
            "make_application_decision",
            "generate_application_draft",
        ]

        ok = (
            res1.decision == "REVIEW"
            and res1.stage == "CLARIFICATION_REQUIRED"
            and res2.decision == "APPLY"
            and res2.application_id == res1.application_id
            and app_count == 1
            and kb_events == 1
            and draft_count == 1
            and r2_tools == expected_r2
        )

        results.append(
            {
                "id": "AG-02",
                "name": "REVIEW -> Clarification -> Run2 APPLY",
                "passed": bool(ok),
                "details": (
                    f"mode={mode}, "
                    f"app_count={app_count}, "
                    f"kb_mutations={kb_events}, "
                    f"tools={r2_tools}"
                ),
            }
        )

        # ============================================================
        # AG-03
        # ============================================================

        db = _fresh_db("applyx_ag03")
        db_paths.append(db)

        kb = db + ".kb"
        kb_dirs.append(kb)

        orch = ApplicationOrchestrator(
            db_path=db,
            kb_dir=kb,
        )

        user = {
            "id": "USER-AG-03",
            "full_name": "Charlie Fresh",
            "degree": "High School Diploma",
            "skills": [
                "HTML"
            ],
            "years_experience": 0,
        }

        opp = {
            "id": "OPP-AG-03",
            "title": "Principal AI Research Scientist",
            "organization": "Gamma Labs",
            "type": "Full-Time",
            "hard_requirements": [
                "PhD in Computer Science"
            ],
        }

        res = _run_scenario(
            orch,
            user,
            opp,
            mode,
        )

        conn = orch._get_db_connection()

        try:
            drafts = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_drafts
                WHERE application_id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

            forbidden = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_events
                WHERE application_id = ?
                  AND tool_name IN (
                      'generate_application_draft',
                      'submit_application'
                  )
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

            stage = conn.execute(
                """
                SELECT current_stage
                FROM applications
                WHERE id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

        finally:
            conn.close()

        ok = (
            res.decision == "SKIP"
            and res.stage == "SKIPPED"
            and stage == "SKIPPED"
            and drafts == 0
            and forbidden == 0
        )

        results.append(
            {
                "id": "AG-03",
                "name": "Hard Failure -> SKIP",
                "passed": bool(ok),
                "details": (
                    f"mode={mode}, "
                    f"stage={stage}, "
                    f"drafts={drafts}, "
                    f"forbidden_tools={forbidden}"
                ),
            }
        )

        # ============================================================
        # AG-04
        # ============================================================

        db = _fresh_db("applyx_ag04")
        db_paths.append(db)

        kb = db + ".kb"
        kb_dirs.append(kb)

        orch = ApplicationOrchestrator(
            db_path=db,
            kb_dir=kb,
        )

        counter = _browser_call_counter(
            orch
        )

        user = {
            "id": "USER-AG-04",
            "full_name": "Dana Dev",
            "degree": "B.Tech CS",
            "skills": [
                "Python",
                "PyTorch",
            ],
            "years_experience": 2,
        }

        opp = {
            "id": "OPP-AG-04",
            "title": "ML Engineer",
            "organization": "Delta Systems",
            "type": "Full-Time",
            "hard_requirements": [
                "Python"
            ],
        }

        res = _run_scenario(
            orch,
            user,
            opp,
            mode,
        )

        rev = orch.create_revised_draft(
            res.application_id,
            updated_fields={
                "full_name": {
                    "value": "Dana Dev",
                    "verified": True,
                }
            },
            updated_answers={
                "why_role": {
                    "value": (
                        "Custom revised motivation statement."
                    )
                }
            },
        )

        appr = orch.approve_draft_version(
            res.application_id,
            2,
        )

        token = set_orchestrator(
            orch
        )

        try:
            old = asyncio.run(
                submit_application(
                    res.application_id,
                    1,
                )
            )

            new = asyncio.run(
                submit_application(
                    res.application_id,
                    2,
                )
            )

        finally:
            reset_orchestrator(
                token
            )

        conn = orch._get_db_connection()

        try:
            row = conn.execute(
                """
                SELECT
                    current_stage,
                    approved_version,
                    submitted_version,
                    confirmation_id,
                    submitted_at
                FROM applications
                WHERE id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()

            draft_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_drafts
                WHERE application_id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

        finally:
            conn.close()

        ok = (
            rev["version"] == 2
            and appr["success"]
            and old.get("success") is False
            and old.get("error_type")
            == "POLICY_GATE_REJECTION"
            and new.get("success") is True
            and new.get("draft_version") == 2
            and counter["calls"] == 1
            and row
            and row[0] == "SUBMITTED"
            and row[1] == 2
            and row[2] == 2
            and row[3]
            and row[4]
            and draft_count == 2
        )

        results.append(
            {
                "id": "AG-04",
                "name": "Version Security V1/V2",
                "passed": bool(ok),
                "details": (
                    f"mode={mode}, "
                    f"browser_calls={counter['calls']}, "
                    f"row={tuple(row) if row else None}"
                ),
            }
        )

        # ============================================================
        # AG-05
        # ============================================================

        db = _fresh_db("applyx_ag05")
        db_paths.append(db)

        kb = db + ".kb"
        kb_dirs.append(kb)

        orch = ApplicationOrchestrator(
            db_path=db,
            kb_dir=kb,
        )

        counter = _browser_call_counter(
            orch
        )

        user = {
            "id": "USER-AG-05",
            "full_name": "Evan Hack",
            "degree": "B.Tech CS",
            "skills": [
                "Python"
            ],
            "years_experience": 1,
        }

        opp = {
            "id": "OPP-AG-05",
            "title": "DevOps Engineer",
            "organization": "Epsilon Inc",
            "type": "Full-Time",
            "hard_requirements": [
                "Python"
            ],
        }

        res = _run_scenario(
            orch,
            user,
            opp,
            mode,
        )

        token = set_orchestrator(
            orch
        )

        try:
            tool_res = asyncio.run(
                submit_application(
                    res.application_id,
                    1,
                )
            )

        finally:
            reset_orchestrator(
                token
            )

        conn = orch._get_db_connection()

        try:
            row = conn.execute(
                """
                SELECT
                    submitted_version,
                    confirmation_id
                FROM applications
                WHERE id = ?
                """,
                (
                    res.application_id,
                ),
            ).fetchone()

            pw = conn.execute(
                """
                SELECT COUNT(*)
                FROM application_events
                WHERE application_id = ?
                  AND event_type LIKE 'STRANDS_%'
                  AND tool_name IS NULL
                  AND stage LIKE '%PLAYWRIGHT%'
                """,
                (
                    res.application_id,
                ),
            ).fetchone()[0]

        finally:
            conn.close()

        ok = (
            tool_res.get("success") is False
            and tool_res.get("error_type")
            == "POLICY_GATE_REJECTION"
            and counter["calls"] == 0
            and row
            and row[0] is None
            and row[1] is None
            and pw == 0
        )

        results.append(
            {
                "id": "AG-05",
                "name": "Unauthorized submission blocked before browser",
                "passed": bool(ok),
                "details": (
                    f"mode={mode}, "
                    f"browser_calls={counter['calls']}, "
                    f"submitted={tuple(row) if row else None}"
                ),
            }
        )

    finally:
        orchestrator_module.DEMO_MODE = previous_demo

        for path in db_paths:
            _remove(path)

        for kb in kb_dirs:
            try:
                import shutil

                shutil.rmtree(
                    kb,
                    ignore_errors=True,
                )

            except Exception:
                pass

    passed = sum(
        bool(item["passed"])
        for item in results
    )

    return {
        "mode": mode,
        "total": len(results),
        "passed": passed,
        "accuracy": (
            passed / len(results) * 100.0
        ),
        "status": (
            "PASS"
            if passed == len(results)
            else "FAIL"
        ),
        "results": results,
    }


def has_real_credentials() -> bool:
    provider = (
        os.getenv(
            "STRANDS_PROVIDER",
            "bedrock",
        )
        .strip()
        .lower()
    )

    if provider == "bedrock":

        if (
            os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            or os.getenv("AWS_ACCESS_KEY_ID")
        ):
            return True

        try:
            import boto3

            return (
                boto3.Session().get_credentials()
                is not None
            )

        except Exception:
            return False

    if provider == "gemini":
        return bool(
            os.getenv(
                "GEMINI_API_KEY"
            )
        )

    if provider == "openai":
        return bool(
            os.getenv(
                "OPENAI_API_KEY"
            )
        )

    return False


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "deterministic",
            "agent",
            "mock",
            "both",
        ],
        default="both",
    )

    args = parser.parse_args()

    exit_code = 0

    if args.mode in {
        "deterministic",
        "both",
    }:

        result_a = run_deterministic_benchmark()

        print(
            json.dumps(
                result_a,
                indent=2,
            )
        )

        exit_code = (
            0
            if result_a["status"] == "PASS"
            else 1
        )

    if args.mode == "mock":

        result_b = run_agent_workflow_benchmark(
            "mock"
        )

        print(
            json.dumps(
                result_b,
                indent=2,
            )
        )

        exit_code = (
            0
            if result_b["status"] == "PASS"
            else 1
        )

    elif args.mode == "agent":

        if not has_real_credentials():

            print(
                "MODE B REAL SKIPPED: "
                "required model credentials are not configured. "
                "Use --mode=mock for the explicit mock backend."
            )

        else:

            result_b = run_agent_workflow_benchmark(
                "real"
            )

            print(
                json.dumps(
                    result_b,
                    indent=2,
                )
            )

            exit_code = (
                0
                if result_b["status"] == "PASS"
                else 1
            )

    elif args.mode == "both":

        if has_real_credentials():

            result_b = run_agent_workflow_benchmark(
                "real"
            )

            print(
                json.dumps(
                    result_b,
                    indent=2,
                )
            )

            exit_code = exit_code or (
                0
                if result_b["status"] == "PASS"
                else 1
            )

        else:

            print(
                "MODE B REAL SKIPPED: "
                "no real-model credentials. "
                "Run --mode=mock for the explicit mock safety harness."
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(
        main()
    )