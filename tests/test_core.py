import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import uuid
from unittest.mock import patch

from core.agent_tools import get_orchestrator, reset_orchestrator, set_orchestrator, submit_application
from core.decision_engine import decide_from_match_analysis
from core.matching_engine import evaluate_candidate_match, parse_raw_requirement
from core.orchestrator import ApplicationOrchestrator
from core.schemas import CanonicalOpportunity, RequirementCategory
from core.state_machine import ALLOWED_TRANSITIONS, ApplicationStatus
from core.strands_agent import (
    AgentInvocationContext,
    TraceSink,
    _MAX_TOOL_CALL_BUDGET,
    _context_invocation,
    clear_active_agent_traces,
    extract_terminal_result,
    get_invocation_traces,
    on_before_tool_call,
    record_agent_trace_event,
    run_strands_application_flow,
)
from core.utils import calculate_profile_completeness, hash_password, verify_password


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.gettempdir(), f"applyx_test_{uuid.uuid4().hex}.db")
        self.kb = os.path.join(tempfile.gettempdir(), f"applyx_test_kb_{uuid.uuid4().hex}")
        self.orch = ApplicationOrchestrator(self.db, self.kb)
        self._orch_token = set_orchestrator(self.orch)

    def tearDown(self):
        reset_orchestrator(self._orch_token)
        try:
            os.remove(self.db)
        except FileNotFoundError:
            pass
        shutil.rmtree(self.kb, ignore_errors=True)


class TestUtils(BaseTest):
    def test_password_round_trip_and_legacy_compatibility(self):
        stored = hash_password("secret")
        self.assertTrue(verify_password("secret", stored))
        self.assertFalse(verify_password("wrong", stored))

    def test_profile_completeness_seven_mandatory(self):
        full = {
            "full_name": "Harsh", "degree": "B.Tech AI & ML", "cgpa": 8.5,
            "work_authorization": "Yes", "graduation_year": 2028,
            "enrollment_status": "CURRENTLY_ENROLLED", "background_check_status": "PASS",
            "skills": ["Python"], "years_experience": 0, "projects": ["P1"],
        }
        result = calculate_profile_completeness(full)
        self.assertEqual(result["mandatory"], 100)
        self.assertEqual(result["optional"], 100)
        self.assertEqual(result["total"], 100)

        unknown = dict(full)
        unknown["work_authorization"] = None
        unknown["enrollment_status"] = None
        unknown["background_check_status"] = None
        result = calculate_profile_completeness(unknown)
        self.assertEqual(result["mandatory"], 57)


class TestMatching(BaseTest):
    def test_parser_experience_range(self):
        req = parse_raw_requirement("1-2 years experience or demonstrable production-level student projects")
        self.assertEqual(req.category, RequirementCategory.EXPERIENCE)
        self.assertEqual(req.operator, "OR_PROJECT")

    def test_missing_enrollment_does_not_infer_from_degree(self):
        profile = {"degree": "B.Tech AI", "skills": ["Python"]}
        opp = CanonicalOpportunity("E1", "Intern", "Org", "Internship", hard_requirements=["Enrolled in B.Tech/STEM"])
        self.assertEqual(evaluate_candidate_match(profile, opp).formal_eligibility, "UNCERTAIN")

    def test_enrollment_and_education_composite(self):
        profile = {"degree": "B.Tech AI & ML", "enrollment_status": "CURRENTLY_ENROLLED"}
        opp = CanonicalOpportunity("E2", "Scholarship", "Org", "Scholarship", hard_requirements=["Enrolled in B.Tech/STEM"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "YES")
        self.assertEqual(len(result.evaluations), 1)
        self.assertEqual(result.evaluations[0].status, "PASS")

    def test_or_skill_requirement(self):
        profile = {"skills": ["Python"]}
        opp = CanonicalOpportunity("E3", "Role", "Org", "Internship", hard_requirements=["C or Python systems programming"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "YES")

    def test_project_or_experience_requirement(self):
        profile = {"years_experience": 0, "projects": []}
        opp = CanonicalOpportunity("E4", "Role", "Org", "Internship", hard_requirements=["1-2 years experience or demonstrable production-level student projects"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "UNCERTAIN")

    def test_missing_skill_is_uncertain_not_fail(self):
        profile = {"degree": "B.Tech CS"}
        opp = CanonicalOpportunity("E5", "Role", "Org", "Internship", hard_requirements=["Python"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "UNCERTAIN")

    def test_phd_hard_failure(self):
        profile = {"degree": "B.Tech AI"}
        opp = CanonicalOpportunity("E6", "Scientist", "Org", "Full-Time", hard_requirements=["PhD in Computer Science or Artificial Intelligence required"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "NO")
        self.assertIn("PhD in Computer Science or Artificial Intelligence required", result.missing_hard)

    def test_background_missing_is_nonblocking(self):
        profile = {"degree": "B.Tech AI", "skills": ["Python"]}
        opp = CanonicalOpportunity("E7", "Role", "Org", "Full-Time", hard_requirements=["Pass Background & Integrity Check", "Python"])
        result = evaluate_candidate_match(profile, opp)
        self.assertEqual(result.formal_eligibility, "YES")


class TestStateAndDrafts(BaseTest):
    def _insert_app(self, app_id, stage="DISCOVERED"):
        now = datetime_now()
        conn = self.orch._get_db_connection()
        conn.execute("INSERT OR REPLACE INTO applications (id,user_id,opportunity_id,current_stage,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)", (app_id, "U", "O", stage, stage, now, now))
        conn.commit(); conn.close()

    def test_allowed_transition(self):
        self.assertIn(ApplicationStatus.NORMALIZED, ALLOWED_TRANSITIONS[ApplicationStatus.DISCOVERED])
        self.assertNotIn(ApplicationStatus.SUBMITTED, ALLOWED_TRANSITIONS[ApplicationStatus.DISCOVERED])

    def test_v1_immutable_and_v2_merge(self):
        app_id = "APP-IMM"
        self._insert_app(app_id, "APPLY")
        user = {"full_name": "A", "degree": "B.Tech CS", "cgpa": 8.0, "skills": ["Python"]}
        opp = CanonicalOpportunity("O", "Role", "Org", "Internship")
        v1 = self.orch.generate_and_persist_draft(app_id, user, opp)
        self.assertEqual(v1["version"], 1)
        snapshot = self.orch.get_draft(app_id, 1)
        self.orch.fsm.transition(app_id, ApplicationStatus.DRAFT_CREATED, reason="Draft")
        self.orch.fsm.transition(app_id, ApplicationStatus.AWAITING_APPROVAL, reason="Await")
        v2 = self.orch.create_revised_draft(app_id, updated_answers={"why_role": {"value": "edited"}})
        self.assertEqual(v2["version"], 2)
        after = self.orch.get_draft(app_id, 1)
        self.assertEqual(snapshot["fields"], after["fields"])
        self.assertEqual(snapshot["answers"], after["answers"])
        self.assertEqual(snapshot["documents"], after["documents"])
        self.assertEqual(snapshot["created_at"], after["created_at"])

    def test_approval_binds_exact_draft(self):
        app_id = "APP-APPROVE"
        self._insert_app(app_id, "APPLY")
        self.orch.generate_and_persist_draft(app_id, {"degree": "B.Tech CS"}, CanonicalOpportunity("O", "Role", "Org", "Internship"))
        self.orch.fsm.transition(app_id, ApplicationStatus.DRAFT_CREATED, reason="Draft")
        self.orch.fsm.transition(app_id, ApplicationStatus.AWAITING_APPROVAL, reason="Await")
        result = self.orch.approve_draft_version(app_id, 1)
        self.assertTrue(result["success"])
        conn = self.orch._get_db_connection(); row = conn.execute("SELECT current_stage,approved_version,approved_draft_id FROM applications WHERE id=?", (app_id,)).fetchone(); conn.close()
        self.assertEqual(row[0], "APPROVED")
        self.assertEqual(row[1], 1)
        self.assertEqual(row[2], "DRAFT-APP-APPROVE-V1")


class TestAgentWorkflow(BaseTest):
    def test_mock_run1_exact_sequence(self):
        user = {"id": "U1", "full_name": "Alice", "degree": "B.Tech CS", "cgpa": 8.8, "skills": ["Python"]}
        opp = {"id": "O1", "title": "Role", "organization": "Org", "type": "Internship", "hard_requirements": ["Python"]}
        self.orch.fetch_user_profile = lambda uid: user
        result = asyncio.run(self.orch.process_opportunity("U1", opp, agent_mode="mock"))
        self.assertEqual(result.decision, "APPLY")
        self.assertEqual(result.stage, "AWAITING_APPROVAL")
        names = [t["tool_name"] for t in result.traces if t.get("event_type") == "BEFORE_TOOL_CALL"]
        self.assertEqual(names, ["get_candidate_profile", "normalize_opportunity", "analyze_candidate_match", "make_application_decision", "generate_application_draft"])

    def test_run2_reuses_application_and_writes_one_clarification(self):
        user = {"id": "U2", "full_name": "Bob", "degree": "B.Tech AI", "skills": ["Python"]}
        opp = {"id": "O2", "title": "Role", "organization": "Org", "type": "Internship", "hard_requirements": ["Python", "Minimum CGPA 8.0 cut-off"]}
        self.orch.fetch_user_profile = lambda uid: user
        result1 = asyncio.run(self.orch.process_opportunity("U2", opp, agent_mode="mock"))
        self.assertEqual(result1.decision, "REVIEW")
        result2 = asyncio.run(self.orch.submit_clarification_and_reevaluate(result1.application_id, "U2", opp, "cgpa", 8.5, clarification_id=result1.clarification_id, agent_mode="mock"))
        self.assertEqual(result2.decision, "APPLY")
        self.assertEqual(result2.application_id, result1.application_id)
        conn = self.orch._get_db_connection()
        apps = conn.execute("SELECT COUNT(*) FROM applications WHERE user_id=? AND opportunity_id=?", ("U2", "O2")).fetchone()[0]
        clar = conn.execute("SELECT COUNT(*) FROM application_events WHERE application_id=? AND event_type='USER_CLARIFICATION'", (result1.application_id,)).fetchone()[0]
        drafts = conn.execute("SELECT COUNT(*) FROM application_drafts WHERE application_id=?", (result1.application_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(apps, 1)
        self.assertEqual(clar, 1)
        self.assertEqual(drafts, 1)

    def test_real_mode_never_falls_back_when_sdk_missing(self):
        with patch("core.strands_agent.STRANDS_AVAILABLE", False):
            ctx = AgentInvocationContext("RUN-1", "APP-REAL", "U", self.db, {"id": "O"})
            token = _context_invocation.set(ctx)
            try:
                with self.assertRaises(RuntimeError):
                    asyncio.run(run_strands_application_flow("U", {"id": "O"}, "APP-REAL", self.orch, agent_mode="real"))
            finally:
                _context_invocation.reset(token)



    def test_approval_rejects_tampered_draft_hash(self):
        app_id = "APP-TAMPER"
        now = datetime_now()
        conn = self.orch._get_db_connection()
        conn.execute("INSERT INTO applications (id,user_id,opportunity_id,current_stage,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)", (app_id,"U","OT","DISCOVERED","DISCOVERED",now,now))
        conn.commit(); conn.close()
        for target in [ApplicationStatus.NORMALIZED, ApplicationStatus.DUPLICATE_CHECKED, ApplicationStatus.ANALYZED, ApplicationStatus.APPLY]:
            self.orch.fsm.transition(app_id, target, reason="test")
        self.orch.generate_and_persist_draft(app_id, {"full_name":"A","degree":"B.Tech CS"}, CanonicalOpportunity("OT","Role","Org","Internship"))
        self.orch.fsm.transition(app_id, ApplicationStatus.DRAFT_CREATED, reason="test")
        self.orch.fsm.transition(app_id, ApplicationStatus.AWAITING_APPROVAL, reason="test")
        conn = self.orch._get_db_connection()
        conn.execute("UPDATE application_drafts SET answers=? WHERE application_id=? AND version=1", ('{"why_role":{"value":"TAMPERED"}}', app_id))
        conn.commit(); conn.close()
        result = self.orch.approve_draft_version(app_id, 1)
        self.assertFalse(result["success"])
        self.assertIn("hash mismatch", result["error"].lower())

    def test_concurrent_duplicate_creation_is_single_row(self):
        import concurrent.futures
        opp = {"id":"CONCURRENT-OPP","title":"Role","organization":"Org","type":"Internship","hard_requirements":["Python"]}
        profile = {"id":"CONCURRENT-USER","full_name":"C","degree":"B.Tech CS","cgpa":8.0,"skills":["Python"]}
        self.orch.fetch_user_profile = lambda uid: profile
        def run():
            return asyncio.run(self.orch.process_opportunity("CONCURRENT-USER", opp, agent_mode="mock"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(lambda _: run(), range(2)))
        conn = self.orch._get_db_connection()
        count = conn.execute("SELECT COUNT(*) FROM applications WHERE user_id=? AND opportunity_id=?", ("CONCURRENT-USER","CONCURRENT-OPP")).fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)
        self.assertEqual(sum(r.decision == "APPLY" for r in results), 1)
        self.assertEqual(sum(r.decision == "DUPLICATE" for r in results), 1)

class TestSubmission(BaseTest):
    def test_approval_version_gate_and_idempotency(self):
        app_id = "APP-SUB"
        now = datetime_now()
        conn = self.orch._get_db_connection()
        conn.execute("INSERT INTO applications (id,user_id,opportunity_id,current_stage,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)", (app_id,"U","O","DISCOVERED","DISCOVERED",now,now))
        conn.commit(); conn.close()
        self.orch.fsm.transition(app_id, ApplicationStatus.NORMALIZED, reason="test")
        self.orch.fsm.transition(app_id, ApplicationStatus.DUPLICATE_CHECKED, reason="test")
        self.orch.fsm.transition(app_id, ApplicationStatus.ANALYZED, reason="test")
        self.orch.fsm.transition(app_id, ApplicationStatus.APPLY, reason="test")
        self.orch.generate_and_persist_draft(app_id, {"full_name":"Bob","degree":"B.Tech AI"}, CanonicalOpportunity("O","Role","Org","Internship"))
        self.orch.fsm.transition(app_id, ApplicationStatus.DRAFT_CREATED, reason="test")
        self.orch.fsm.transition(app_id, ApplicationStatus.AWAITING_APPROVAL, reason="test")
        self.assertTrue(self.orch.approve_draft_version(app_id, 1)["success"])
        with patch("core.orchestrator.DEMO_MODE", True):
            result = asyncio.run(submit_application(app_id, 1))
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("draft_version"), 1)

    def test_unapproved_submission_is_blocked_before_browser(self):
        app_id = "APP-GATE"
        now = datetime_now()
        conn = self.orch._get_db_connection()
        conn.execute("INSERT INTO applications (id,user_id,opportunity_id,current_stage,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)", (app_id,"U","O","AWAITING_APPROVAL","AWAITING_APPROVAL",now,now))
        conn.execute("INSERT INTO application_drafts (id,application_id,version,fields,answers,documents,created_at) VALUES (?,?,?,?,?,?,?)", ("D1",app_id,1,"{}","{}","[]",now))
        conn.commit(); conn.close()
        with patch.object(self.orch, "_execute_browser_submission", side_effect=AssertionError("browser must not run")):
            result = asyncio.run(submit_application(app_id, 1))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "POLICY_GATE_REJECTION")


class TestTrace(BaseTest):
    def setUp(self):
        super().setUp()
        clear_active_agent_traces()

    def test_trace_queue_db_and_sequence(self):
        sink = TraceSink.get_instance()
        q = sink.register_live_queue("APP-T", user_id="U")
        ctx = AgentInvocationContext("RUN-T", "APP-T", "U", self.db, {"id": "O"})
        rec1 = record_agent_trace_event("BEFORE_TOOL_CALL", {}, ctx, "normalize_opportunity")
        rec2 = record_agent_trace_event("AFTER_TOOL_CALL", {}, ctx, "normalize_opportunity")
        events = sink.drain_live_queue(q)
        sink.unregister_live_queue("APP-T", q)
        self.assertEqual([e["sequence_number"] for e in events], [rec1["sequence_number"], rec2["sequence_number"]])
        self.assertEqual(events[0]["sequence_number"], 1)
        self.assertEqual(events[1]["sequence_number"], 2)
        conn = self.orch._get_db_connection(); count = conn.execute("SELECT COUNT(*) FROM application_events WHERE application_id='APP-T' AND run_id='RUN-T'").fetchone()[0]; conn.close()
        self.assertEqual(count, 2)

    def test_tool_budget_hook_marks_cancellation(self):
        ctx = AgentInvocationContext("RUN-T", "APP-BUDGET", "U", self.db, {"id": "O"})
        token = _context_invocation.set(ctx)
        try:
            class Event:
                cancel_tool = False
                selected_tool = type("Tool", (), {"name": "x"})()
            for _ in range(_MAX_TOOL_CALL_BUDGET):
                on_before_tool_call(Event())
            event = Event()
            on_before_tool_call(event)
            self.assertTrue(event.cancel_tool)
            self.assertEqual(ctx.cancellation_reason, "TOOL_LIMIT")
        finally:
            _context_invocation.reset(token)

    def test_terminal_result_adapter(self):
        obj = type("R", (), {"stop_reason": "end_turn"})()
        self.assertIs(extract_terminal_result(obj), obj)


def datetime_now():
    from datetime import datetime
    return datetime.now().isoformat()


if __name__ == "__main__":
    unittest.main(verbosity=2)
