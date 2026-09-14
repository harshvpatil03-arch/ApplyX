"""
core/orchestrator.py
Central ApplyX application lifecycle authority.

Responsibilities:
- Canonical opportunity normalization
- Application duplicate protection
- FSM-owned lifecycle progression
- Knowledge Base persistence and clarification provenance
- Immutable draft creation/revision
- Human approval/version binding
- Deterministic Playwright submission with idempotency

The Strands layer may reason and select tools, but it does not own lifecycle state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import CanonicalOpportunity, MatchAnalysisResult
from core.state_machine import ApplicationStatus, AuthoritativeStateMachine
from core.matching_engine import evaluate_candidate_match

logger = logging.getLogger(__name__)

MAX_CLARIFICATION_ATTEMPTS = 2
ALLOWED_KB_FIELDS = {
    "full_name",
    "email",
    "phone",
    "location",
    "degree",
    "college_name",
    "education_history",
    "cgpa",
    "graduation_year",
    "enrollment_status",
    "work_authorization",
    "background_check_status",
    "years_experience",
    "work_experience",
    "skills",
    "projects",
    "linkedin_url",
    "github_url",
    "portfolio_url",
    "profile_photo_path",
    "resume_path",
    "resume_name",
}

DEMO_MODE = os.getenv("DEMO_MODE", "false").strip().lower() in {"true", "1", "yes"}


def _extract_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_evidence_coverage(match_analysis: Any) -> float:
    return float(_extract_field(match_analysis, "evidence_coverage", 0.0) or 0.0)


@dataclass
class BrowserExecutionSuccess:
    success: bool = True
    application_id: str = ""
    draft_version: int = 1
    submitted: bool = True
    confirmation_id: str = ""
    execution_mode: str = "REAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BrowserExecutionFailure:
    success: bool = False
    submitted: bool = False
    error_type: str = "FIELD_NOT_FOUND"
    error: str = ""
    execution_mode: str = "REAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorResult:
    application_id: str
    opportunity_id: str
    stage: str
    decision: str
    confidence: float = 0.0
    fit_score: int = 0
    evidence_coverage: float = 0.0
    risk: str = "UNKNOWN"
    draft_version: Optional[int] = None
    confirmation_number: Optional[str] = None
    reason: Optional[str] = None
    clarification_prompt: Optional[str] = None
    clarification_id: Optional[str] = None
    clarification_attempts: int = 0
    logs: List[str] = field(default_factory=list)
    match_analysis: Optional[MatchAnalysisResult | Dict[str, Any]] = None
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if hasattr(self.match_analysis, "to_dict"):
            data["match_analysis"] = self.match_analysis.to_dict()
        return data


class ApplicationOrchestrator:
    _schema_process_lock = threading.RLock()
    _schema_initialized_paths: set[str] = set()

    def __init__(self, db_path: str = "agent_applications.db", kb_dir: Optional[str] = None) -> None:
        self.db_path = db_path
        self.kb_dir = kb_dir or os.getenv("APPLYX_KB_DIR", os.path.join("data", "users"))
        self.fsm = AuthoritativeStateMachine(db_path=db_path)
        self._ensure_schema_initialized()

    def _get_db_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=60.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA foreign_keys=ON")
        # WAL permits readers while a short writer transaction is in progress.
        # Ignore failures here because a read-only filesystem may reject the setting.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        return conn

    def _ensure_schema_initialized(self) -> None:
        """Create/migrate the single canonical SQLite schema used by ApplyX.

        Streamlit reruns can construct multiple orchestrators in one process. Schema
        reconciliation is therefore performed once per database path per process,
        under a process-local lock, rather than issuing UPDATEs on every rerun.
        """
        canonical_path = os.path.abspath(self.db_path)
        with self._schema_process_lock:
            if canonical_path in self._schema_initialized_paths:
                return
            os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
            conn = self._get_db_connection()
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS applications (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        opportunity_id TEXT NOT NULL,
                        current_stage TEXT NOT NULL,
                        status TEXT NOT NULL,
                        approved_version INTEGER DEFAULT NULL,
                        submitted_version INTEGER DEFAULT NULL,
                        approved_draft_id TEXT DEFAULT NULL,
                        confirmation_id TEXT DEFAULT NULL,
                        error_message TEXT DEFAULT NULL,
                        submitted_at TEXT DEFAULT NULL,
                        clarification_attempts INTEGER NOT NULL DEFAULT 0,
                        clarification_id TEXT DEFAULT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                self._ensure_columns(
                    cur,
                    "applications",
                    {
                        "clarification_attempts": "INTEGER NOT NULL DEFAULT 0",
                        "clarification_id": "TEXT DEFAULT NULL",
                    },
                )
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_user_opportunity ON applications(user_id, opportunity_id)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS application_drafts (
                        id TEXT PRIMARY KEY,
                        application_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        fields TEXT NOT NULL,
                        answers TEXT NOT NULL,
                        documents TEXT NOT NULL DEFAULT '[]',
                        payload_hash TEXT DEFAULT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                self._ensure_columns(
                    cur,
                    "application_drafts",
                    {
                        "documents": "TEXT NOT NULL DEFAULT '[]'",
                        "payload_hash": "TEXT DEFAULT NULL",
                    },
                )
                self._reconcile_duplicate_versions(cur)
                self._reconcile_draft_hashes(cur)
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_app_draft_version ON application_drafts(application_id, version)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS application_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        application_id TEXT NOT NULL,
                        run_id TEXT NOT NULL DEFAULT 'RUN-1',
                        sequence_number INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        stage TEXT,
                        tool_name TEXT,
                        event_payload TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                self._ensure_columns(
                    cur,
                    "application_events",
                    {
                        "run_id": "TEXT DEFAULT 'RUN-1'",
                        "sequence_number": "INTEGER DEFAULT NULL",
                        "event_type": "TEXT DEFAULT 'EVENT'",
                        "tool_name": "TEXT DEFAULT NULL",
                    },
                )
                self._reconcile_event_sequences(cur)
                cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_trace_sequence ON application_events(application_id, run_id, sequence_number)")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS opportunities (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        organization TEXT NOT NULL,
                        type TEXT NOT NULL,
                        requirements TEXT,
                        source_url TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                self._ensure_columns(cur, "opportunities", {"source_url": "TEXT DEFAULT NULL"})
                conn.commit()
            finally:
                conn.close()
            self._schema_initialized_paths.add(canonical_path)

    @staticmethod
    def _ensure_columns(cur: sqlite3.Cursor, table: str, required: Dict[str, str]) -> None:
        existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, definition in required.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _reconcile_duplicate_versions(cur: sqlite3.Cursor) -> None:
        rows = cur.execute(
            """
            SELECT application_id, version, COUNT(*) AS cnt
            FROM application_drafts
            GROUP BY application_id, version
            HAVING cnt > 1
            """
        ).fetchall()
        for row in rows:
            keep = cur.execute(
                """
                SELECT rowid FROM application_drafts
                WHERE application_id = ? AND version = ?
                ORDER BY created_at ASC, rowid ASC LIMIT 1
                """,
                (row[0], row[1]),
            ).fetchone()
            if keep:
                cur.execute(
                    "DELETE FROM application_drafts WHERE application_id = ? AND version = ? AND rowid <> ?",
                    (row[0], row[1], keep[0]),
                )

    @staticmethod
    def _draft_payload_hash(fields: Dict[str, Any], answers: Dict[str, Any], documents: List[Dict[str, Any]]) -> str:
        import hashlib
        canonical = json.dumps(
            {"fields": fields, "answers": answers, "documents": documents},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _reconcile_draft_hashes(cls, cur: sqlite3.Cursor) -> None:
        rows = cur.execute(
            "SELECT id, fields, answers, documents, payload_hash FROM application_drafts"
        ).fetchall()
        for row in rows:
            if row[4]:
                continue
            try:
                fields = json.loads(row[1] or "{}")
                answers = json.loads(row[2] or "{}")
                documents = json.loads(row[3] or "[]")
                payload_hash = cls._draft_payload_hash(fields, answers, documents)
                cur.execute(
                    "UPDATE application_drafts SET payload_hash = ? WHERE id = ?",
                    (payload_hash, row[0]),
                )
            except Exception:
                continue

    @staticmethod
    def _reconcile_event_sequences(cur: sqlite3.Cursor) -> None:
        # Backfill missing run_id / event_type / sequence_number and compact duplicates.
        cur.execute("UPDATE application_events SET run_id = COALESCE(NULLIF(run_id, ''), 'RUN-1')")
        rows = cur.execute(
            "SELECT id, application_id, run_id, sequence_number FROM application_events ORDER BY application_id, run_id, created_at, id"
        ).fetchall()
        counters: Dict[Tuple[str, str], int] = {}
        for row in rows:
            key = (row[1], row[2])
            counters[key] = counters.get(key, 0) + 1
            desired = counters[key]
            if row[3] != desired:
                cur.execute("UPDATE application_events SET sequence_number = ? WHERE id = ?", (desired, row[0]))
        cur.execute("UPDATE application_events SET event_type = COALESCE(NULLIF(event_type, ''), COALESCE(stage, 'EVENT'))")

    @staticmethod
    def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="applyx_", suffix=".json", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _next_event_sequence(self, cur: sqlite3.Cursor, app_id: str, run_id: str) -> int:
        row = cur.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM application_events WHERE application_id = ? AND run_id = ?",
            (app_id, run_id),
        ).fetchone()
        return int(row[0]) if row else 1

    def log_audit_event(
        self,
        app_id: str,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: str = "RUN-1",
        tool_name: Optional[str] = None,
        sequence_number: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist exactly one immutable event; never replace an existing sequence."""
        metadata = metadata or {}
        now_iso = datetime.now().isoformat()
        r_id = str(run_id or metadata.get("run_id") or "RUN-1")
        t_name = tool_name or metadata.get("tool_name") or metadata.get("tool")
        e_type = event_type or metadata.get("event_type") or stage

        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            seq = int(sequence_number) if sequence_number is not None else self._next_event_sequence(cur, app_id, r_id)
            payload = dict(metadata)
            payload.setdefault("event_type", e_type)
            payload.setdefault("stage", stage)
            payload.setdefault("run_id", r_id)
            payload.setdefault("sequence_number", seq)
            cur.execute(
                """
                INSERT INTO application_events (
                    application_id, run_id, sequence_number, event_type,
                    stage, tool_name, event_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (app_id, r_id, seq, e_type, stage, t_name, json.dumps(payload, default=str), now_iso),
            )
            conn.commit()
            return {
                "application_id": app_id,
                "run_id": r_id,
                "sequence_number": seq,
                "event_type": e_type,
                "stage": stage,
                "tool_name": t_name,
                "timestamp": now_iso,
                "details": payload,
            }
        except Exception:
            conn.rollback()
            logger.exception("Audit event logging failed for %s", app_id)
            raise
        finally:
            conn.close()

    def get_draft(self, application_id: str, version: int = 1) -> Optional[Dict[str, Any]]:
        conn = self._get_db_connection()
        try:
            row = conn.execute(
                "SELECT id, application_id, version, fields, answers, documents, payload_hash, created_at FROM application_drafts WHERE application_id = ? AND version = ?",
                (application_id, version),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def fail_application(self, application_id: str, reason: str, error_code: Optional[str] = None, run_id: str = "RUN-1") -> bool:
        """Centralized failure entry point with rollback-safe FSM transition."""
        now_iso = datetime.now().isoformat()
        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            err_msg = f"[{error_code}] {reason}" if error_code else reason
            cur.execute(
                """
                UPDATE applications
                SET submitted_version = NULL,
                    confirmation_id = NULL,
                    submitted_at = NULL,
                    error_message = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (err_msg, now_iso, application_id),
            )
            if cur.rowcount != 1:
                conn.rollback()
                return False

            ok, message = self.fsm.transition(
                application_id,
                ApplicationStatus.FAILED,
                reason=reason,
                metadata={"run_id": run_id},
                external_conn=conn,
            )
            if not ok:
                conn.rollback()
                logger.error("fail_application transition rejected: %s", message)
                return False

            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Failure transaction failed for %s", application_id)
            return False
        finally:
            conn.close()

        try:
            self.log_audit_event(application_id, "APPLICATION_FAILED", {"reason": reason, "error_code": error_code}, run_id=run_id)
        except Exception:
            logger.exception("Failure audit event could not be persisted for %s", application_id)
        return True

    def generate_safe_application_id(self) -> str:
        return f"APP-{uuid.uuid4().hex[:12].upper()}"

    def fetch_user_profile(self, user_id: str) -> Dict[str, Any]:
        os.makedirs(self.kb_dir, exist_ok=True)
        user_file = os.path.join(self.kb_dir, f"{user_id}.json")
        if os.path.exists(user_file):
            try:
                with open(user_file, "r", encoding="utf-8") as handle:
                    profile = json.load(handle)
                if isinstance(profile, dict):
                    profile.setdefault("id", user_id)
                    return profile
            except Exception as exc:
                logger.error("Failed to read profile %s: %s", user_file, exc)

        # Explicit opt-in seed profile only; never silently fabricate candidate facts.
        seed_path = os.getenv("APPLYX_SEED_PROFILE_PATH", "user.json")
        use_seed = os.getenv("APPLYX_USE_SEED_PROFILE", "false").lower() in {"true", "1", "yes"}
        if use_seed and os.path.exists(seed_path):
            try:
                with open(seed_path, "r", encoding="utf-8") as handle:
                    seed = json.load(handle)
                if isinstance(seed, dict) and seed.get("id") == user_id:
                    self._atomic_write_json(user_file, seed)
                    return seed
            except Exception as exc:
                logger.error("Failed to import seed profile: %s", exc)

        profile = {
            "id": user_id,
            "full_name": None,
            "email": None,
            "phone": None,
            "location": None,
            "degree": None,
            "college_name": None,
            "education_history": [],
            "cgpa": None,
            "graduation_year": None,
            "enrollment_status": None,
            "work_authorization": None,
            "background_check_status": None,
            "years_experience": None,
            "work_experience": [],
            "skills": [],
            "projects": [],
            "linkedin_url": None,
            "github_url": None,
            "portfolio_url": None,
            "profile_photo_path": None,
            "resume_path": None,
            "resume_name": None,
            "evidence_ledger": {},
        }
        self._atomic_write_json(user_file, profile)
        return profile

    def update_user_kb(
        self,
        user_id: str,
        key: str,
        value: Any,
        application_id: Optional[str] = None,
        clarification_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update one whitelisted KB field; clarification writes are serialized and audited atomically."""
        if key not in ALLOWED_KB_FIELDS:
            raise ValueError(f"Unsupported Knowledge Base field: '{key}'. Allowed fields: {sorted(ALLOWED_KB_FIELDS)}")

        if application_id and clarification_id and not re.fullmatch(r"CLR-[0-9a-fA-F-]{36}", clarification_id):
            raise ValueError("Invalid clarification_id format.")

        profile = self.fetch_user_profile(user_id)
        old_value = profile.get(key)
        profile[key] = value
        user_file = os.path.join(self.kb_dir, f"{user_id}.json")

        if not application_id:
            self._atomic_write_json(user_file, profile)
            return profile

        clr_id = clarification_id or f"CLR-{uuid.uuid4()}"
        if not re.fullmatch(r"CLR-[0-9a-fA-F-]{36}", clr_id):
            raise ValueError("Invalid clarification_id format.")

        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            app = conn.execute(
                "SELECT user_id, current_stage, clarification_attempts FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
            if not app:
                conn.rollback()
                raise ValueError(f"Application '{application_id}' not found.")
            if app[0] != user_id:
                conn.rollback()
                raise ValueError("User/application ownership mismatch.")
            if app[1] != ApplicationStatus.CLARIFICATION_REQUIRED.value:
                conn.rollback()
                raise ValueError(f"Clarification update is not allowed from state '{app[1]}'.")
            attempts = int(app[2] or 0)
            if attempts >= MAX_CLARIFICATION_ATTEMPTS:
                conn.rollback()
                raise ValueError("MAX_CLARIFICATION_ATTEMPTS_EXCEEDED")

            # Persist the KB atomically with the corresponding audit event. If the
            # DB transaction fails, restore the previous file snapshot.
            self._atomic_write_json(user_file, profile)
            now_iso = datetime.now().isoformat()
            payload = {
                "clarification_id": clr_id,
                "field": key,
                "old_value": old_value,
                "new_value": value,
                "source": "USER_CLARIFICATION",
                "timestamp": now_iso,
            }
            seq = self._next_event_sequence(conn.cursor(), application_id, "RUN-2")
            conn.execute(
                """
                INSERT INTO application_events (
                    application_id, run_id, sequence_number, event_type, stage,
                    tool_name, event_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id, "RUN-2", seq, "USER_CLARIFICATION",
                    "USER_CLARIFICATION", None, json.dumps(payload, default=str), now_iso,
                ),
            )
            conn.execute(
                """
                UPDATE applications
                SET clarification_attempts = clarification_attempts + 1,
                    clarification_id = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (clr_id, now_iso, application_id),
            )
            conn.commit()
            return profile
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            # Restore the old candidate profile snapshot after a failed DB commit.
            restored = dict(profile)
            restored[key] = old_value
            try:
                self._atomic_write_json(user_file, restored)
            except Exception:
                logger.exception("Failed to restore KB after clarification transaction failure.")
            raise
        finally:
            conn.close()

    def normalize_opportunity(self, raw_opp: Any) -> CanonicalOpportunity:
        """Normalize dictionaries/objects/IDs without inventing requirement facts."""
        if isinstance(raw_opp, CanonicalOpportunity):
            return raw_opp

        if isinstance(raw_opp, str):
            opportunity_id = raw_opp.strip()
            conn = self._get_db_connection()
            try:
                row = conn.execute("SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)).fetchone()
            finally:
                conn.close()
            if not row:
                return CanonicalOpportunity(id=opportunity_id, title="Target Role", organization="Target Organization", type="Internship")
            payload = dict(row)
            requirements = json.loads(payload.get("requirements") or "{}")
            if isinstance(requirements, dict):
                hard = requirements.get("hard", []) or []
                preferred = requirements.get("preferred", []) or []
                conditional = requirements.get("conditional", []) or []
                all_requirements = hard + preferred + conditional
            else:
                all_requirements = requirements if isinstance(requirements, list) else []
                hard = [r for r in all_requirements if not any(w in str(r).lower() for w in ("preferred", "nice to have", "bonus"))]
                preferred = [r for r in all_requirements if str(r) not in hard]
                conditional = []
            return CanonicalOpportunity(
                id=opportunity_id,
                title=payload.get("title") or "Target Role",
                organization=payload.get("organization") or "Target Organization",
                type=payload.get("type") or "Internship",
                requirements=all_requirements,
                hard_requirements=hard,
                preferred_requirements=preferred,
                conditional_requirements=conditional,
                source_url=payload.get("source_url"),
            )

        if hasattr(raw_opp, "to_dict") and callable(raw_opp.to_dict):
            raw_opp = raw_opp.to_dict()
        if not isinstance(raw_opp, dict):
            raise TypeError("Opportunity must be a dict, CanonicalOpportunity, or opportunity ID string.")

        opportunity_id = str(raw_opp.get("id") or raw_opp.get("opportunity_id") or f"OPP-{uuid.uuid4().hex[:10].upper()}").strip()
        title = str(raw_opp.get("title") or raw_opp.get("role") or "Target Role").strip()
        organization = str(raw_opp.get("organization") or raw_opp.get("company") or "Target Organization").strip()
        opp_type = str(raw_opp.get("type") or "Internship").strip()

        def as_list(value: Any) -> List[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                    if isinstance(decoded, list):
                        return [str(v).strip() for v in decoded if str(v).strip()]
                except Exception:
                    pass
                return [value.strip()] if value.strip() else []
            return []

        raw_requirements = as_list(raw_opp.get("requirements"))
        hard = as_list(raw_opp.get("hard_requirements"))
        preferred = as_list(raw_opp.get("preferred_requirements"))
        conditional = as_list(raw_opp.get("conditional_requirements"))

        mandatory = raw_opp.get("mandatory_criteria")
        if isinstance(mandatory, dict):
            hard.extend(as_list(mandatory.get("required_skills")))
            if mandatory.get("min_cgpa") is not None:
                hard.append(f"Minimum CGPA {mandatory['min_cgpa']} cut-off")

        if not hard and not preferred and not conditional and raw_requirements:
            hard = [r for r in raw_requirements if not any(w in r.lower() for w in ("preferred", "nice to have", "bonus", "optional"))]
            preferred = [r for r in raw_requirements if r not in hard]
        if not raw_requirements:
            raw_requirements = hard + preferred + conditional

        return CanonicalOpportunity(
            id=opportunity_id,
            title=title,
            organization=organization,
            type=opp_type,
            location=raw_opp.get("location"),
            deadline=raw_opp.get("deadline"),
            requirements=raw_requirements,
            hard_requirements=hard,
            preferred_requirements=preferred,
            conditional_requirements=conditional,
            source=raw_opp.get("source", "Direct"),
            source_url=raw_opp.get("source_url"),
        )

    def check_duplicate(self, user_id: str, opportunity_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_db_connection()
        try:
            row = conn.execute(
                """
                SELECT id, current_stage, status, approved_version, submitted_version, confirmation_id
                FROM applications WHERE user_id = ? AND opportunity_id = ? ORDER BY rowid DESC LIMIT 1
                """,
                (user_id, opportunity_id),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def validate_claims(self, answers: Dict[str, Any], user_profile: Dict[str, Any]) -> Tuple[bool, str]:
        """Conservatively validate generated claims against the current KB."""
        combined_text = " ".join(str(v.get("value", "")).lower() for v in answers.values() if isinstance(v, dict))
        degree = str(user_profile.get("degree") or "").lower()
        exp = user_profile.get("years_experience")
        try:
            exp_value = int(exp) if exp is not None else 0
        except (TypeError, ValueError):
            exp_value = 0
        if "phd" in combined_text and "phd" not in degree and "doctorate" not in degree:
            return False, "Generated draft contained an unsupported PhD claim."
        if re.search(r"\b10\+?\s*years?\b", combined_text) and exp_value < 10:
            return False, "Generated draft contained an unsupported 10+ years experience claim."
        return True, "Generated claims are grounded in available candidate facts."

    def _draft_payload(self, user: Dict[str, Any], opp: CanonicalOpportunity) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        fields: Dict[str, Any] = {}
        for field_name in sorted(ALLOWED_KB_FIELDS):
            value = user.get(field_name)
            fields[field_name] = {
                "value": value,
                "source": f"user_profile.{field_name}" if value is not None else "unverified",
                "verified": value is not None,
            }

        degree = user.get("degree")
        college_name = user.get("college_name")
        projects = user.get("projects") or []
        skills = user.get("skills") or []
        work_experience = user.get("work_experience") or []
        facts: List[str] = []
        if degree and college_name:
            facts.append(f"my {degree} studies at {college_name}")
        elif degree:
            facts.append(f"my academic background in {degree}")
        elif college_name:
            facts.append(f"my studies at {college_name}")
        if work_experience:
            companies = []
            for item in work_experience[:3]:
                if isinstance(item, dict) and item.get("company"):
                    companies.append(str(item["company"]))
            if companies:
                facts.append(f"professional experience at {', '.join(companies)}")
        if projects:
            facts.append(f"projects such as {', '.join(map(str, projects[:3]))}")
        if skills:
            facts.append(f"skills including {', '.join(map(str, skills[:5]))}")

        if facts:
            answer_text = f"I am interested in {opp.title} at {opp.organization} because {' and '.join(facts)} provide a relevant foundation for the role."
        else:
            answer_text = f"I am interested in the {opp.title} opportunity at {opp.organization} and would value the opportunity to contribute while continuing to develop relevant skills."

        evidence_keys = {"degree", "college_name", "work_experience", "projects", "skills"}
        answers = {
            "why_role": {
                "value": answer_text,
                "source": "evidence_grounded_generator",
                "evidence_ids": [f"user_profile.{key}" for key, value in user.items() if key in evidence_keys and value][:5],
            }
        }
        documents: List[Dict[str, Any]] = []
        resume_path = user.get("resume_path")
        resume_name = user.get("resume_name") or (os.path.basename(resume_path) if resume_path else None)
        if resume_path and os.path.exists(str(resume_path)):
            documents.append({"type": "resume", "name": resume_name or "Resume", "path": str(resume_path), "source": "user_profile.resume_path"})
        return fields, answers, documents

    def generate_and_persist_draft(self, application_id: str, user: Dict[str, Any], opp: CanonicalOpportunity) -> Dict[str, Any]:
        """Idempotently create immutable Draft V1."""
        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id FROM application_drafts WHERE application_id = ? AND version = 1",
                (application_id,),
            ).fetchone()
            if existing:
                conn.rollback()
                return {"status": "ALREADY_EXISTS", "application_id": application_id, "draft_id": existing[0], "version": 1}

            fields, answers, documents = self._draft_payload(user, opp)
            ok, reason = self.validate_claims(answers, user)
            if not ok:
                raise ValueError(reason)

            now_iso = datetime.now().isoformat()
            draft_id = f"DRAFT-{application_id}-V1"
            payload_hash = self._draft_payload_hash(fields, answers, documents)
            conn.execute(
                """
                INSERT INTO application_drafts (id, application_id, version, fields, answers, documents, payload_hash, created_at)
                VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    application_id,
                    json.dumps(fields, default=str),
                    json.dumps(answers, default=str),
                    json.dumps(documents),
                    payload_hash,
                    now_iso,
                ),
            )
            conn.execute(
                """
                UPDATE applications
                SET approved_version = NULL, approved_draft_id = NULL, updated_at = ?
                WHERE id = ?
                """,
                (now_iso, application_id),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            existing = self.get_draft(application_id, 1)
            if existing:
                return {"status": "ALREADY_EXISTS", "application_id": application_id, "draft_id": existing["id"], "version": 1}
            raise
        finally:
            conn.close()

        return {
            "status": "DRAFT_CREATED",
            "application_id": application_id,
            "draft_id": draft_id,
            "version": 1,
            "approved_version": None,
        }

    @staticmethod
    def _deep_merge(old: Any, new: Any) -> Any:
        if isinstance(old, dict) and isinstance(new, dict):
            merged = dict(old)
            for key, value in new.items():
                merged[key] = ApplicationOrchestrator._deep_merge(merged[key], value) if key in merged else value
            return merged
        return new

    def create_revised_draft(
        self,
        application_id: str,
        updated_fields: Optional[Dict[str, Any]] = None,
        updated_answers: Optional[Dict[str, Any]] = None,
        updated_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create V(n+1) without mutating any prior version and invalidate approval."""
        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            app_row = conn.execute("SELECT current_stage FROM applications WHERE id = ?", (application_id,)).fetchone()
            if not app_row:
                raise ValueError(f"Application '{application_id}' not found.")
            if app_row[0] not in {
                ApplicationStatus.AWAITING_APPROVAL.value,
                ApplicationStatus.APPROVED.value,
                ApplicationStatus.DRAFT_CREATED.value,
            }:
                raise ValueError(f"Draft revision is not allowed from application state '{app_row[0]}'.")

            row = conn.execute(
                "SELECT version, fields, answers, documents FROM application_drafts WHERE application_id = ? ORDER BY version DESC LIMIT 1",
                (application_id,),
            ).fetchone()
            if not row:
                raise ValueError("Cannot revise an application with no existing draft.")

            previous_version = int(row[0])
            next_version = previous_version + 1
            old_fields = json.loads(row[1])
            old_answers = json.loads(row[2])
            old_documents = json.loads(row[3] or "[]")

            merged_fields = self._deep_merge(old_fields, updated_fields or {})
            merged_answers = self._deep_merge(old_answers, updated_answers or {})
            merged_documents = updated_documents if updated_documents is not None else old_documents
            now_iso = datetime.now().isoformat()
            draft_id = f"DRAFT-{application_id}-V{next_version}"
            payload_hash = self._draft_payload_hash(merged_fields, merged_answers, merged_documents)

            conn.execute(
                """
                INSERT INTO application_drafts (id, application_id, version, fields, answers, documents, payload_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft_id,
                    application_id,
                    next_version,
                    json.dumps(merged_fields, default=str),
                    json.dumps(merged_answers, default=str),
                    json.dumps(merged_documents, default=str),
                    payload_hash,
                    now_iso,
                ),
            )
            conn.execute(
                """
                UPDATE applications
                SET approved_version = NULL,
                    approved_draft_id = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now_iso, application_id),
            )

            target = ApplicationStatus.AWAITING_APPROVAL
            ok, message = self.fsm.transition(
                application_id,
                target,
                reason=f"Revised immutable draft V{next_version} created",
                metadata={"run_id": "RUN-1"},
                external_conn=conn,
            )
            if not ok:
                raise RuntimeError(message)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        self.log_audit_event(application_id, "DRAFT_REVISED", {"new_version": next_version, "draft_id": draft_id}, run_id="HUMAN")
        return {"success": True, "draft_id": draft_id, "version": next_version, "status": ApplicationStatus.AWAITING_APPROVAL.value}

    def approve_draft_version(self, application_id: str, version: int) -> Dict[str, Any]:
        """Atomically bind the exact immutable draft version to human approval."""
        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            app = conn.execute("SELECT current_stage FROM applications WHERE id = ?", (application_id,)).fetchone()
            if not app:
                conn.rollback()
                return {"success": False, "error": "Application not found."}
            if app[0] != ApplicationStatus.AWAITING_APPROVAL.value:
                conn.rollback()
                return {"success": False, "error": f"Application must be AWAITING_APPROVAL, currently {app[0]}."}

            draft = conn.execute(
                "SELECT id, fields, answers, documents, payload_hash FROM application_drafts WHERE application_id = ? AND version = ?",
                (application_id, version),
            ).fetchone()
            if not draft:
                conn.rollback()
                return {"success": False, "error": f"Draft V{version} not found."}

            # Integrity validation before authorization.
            try:
                fields = json.loads(draft[1])
                answers = json.loads(draft[2])
                documents = json.loads(draft[3] or "[]")
                if not isinstance(fields, dict) or not isinstance(answers, dict) or not isinstance(documents, list):
                    raise ValueError("draft payload has invalid container types")
                actual_hash = self._draft_payload_hash(fields, answers, documents)
                if draft[4] != actual_hash:
                    raise ValueError("draft payload hash mismatch")
            except Exception as exc:
                conn.rollback()
                return {"success": False, "error": f"Draft payload is corrupt: {exc}"}

            now_iso = datetime.now().isoformat()
            conn.execute(
                "UPDATE applications SET approved_version = ?, approved_draft_id = ?, updated_at = ? WHERE id = ?",
                (version, draft[0], now_iso, application_id),
            )
            ok, message = self.fsm.transition(
                application_id,
                ApplicationStatus.APPROVED,
                reason=f"Human approved Draft V{version}",
                external_conn=conn,
            )
            if not ok:
                conn.rollback()
                return {"success": False, "error": message}
            conn.commit()
        except Exception as exc:
            conn.rollback()
            return {"success": False, "error": str(exc)}
        finally:
            conn.close()

        self.log_audit_event(application_id, "HUMAN_APPROVED", {"approved_version": version, "draft_id": draft[0]}, run_id="HUMAN")
        return {"success": True, "application_id": application_id, "approved_version": version, "draft_id": draft[0]}

    def pre_submission_policy_gate(self, application_id: str, version: int) -> Tuple[bool, List[str]]:
        """Authoritative pre-dispatch security gate."""
        errors: List[str] = []
        conn = self._get_db_connection()
        try:
            app = conn.execute(
                "SELECT current_stage, approved_version, approved_draft_id, submitted_version, confirmation_id FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
            if not app:
                return False, ["Application record missing in database."]

            stage = app[0]
            if stage != ApplicationStatus.APPROVED.value:
                errors.append(f"Security Gate: application state is '{stage}', required 'APPROVED'.")
            if app[1] is None:
                errors.append("Security Gate: human approval is missing.")
            elif int(app[1]) != int(version):
                errors.append(f"Security Gate: approved version is V{app[1]}, attempted V{version}.")

            draft = conn.execute(
                "SELECT id, fields, answers, documents, payload_hash FROM application_drafts WHERE application_id = ? AND version = ?",
                (application_id, version),
            ).fetchone()
            if not draft:
                errors.append(f"Security Gate: Draft V{version} does not exist.")
            else:
                if not app[2] or draft[0] != app[2]:
                    errors.append("Security Gate: approved_draft_id does not match the requested immutable draft.")
                try:
                    fields = json.loads(draft[1])
                    answers = json.loads(draft[2])
                    documents = json.loads(draft[3] or "[]")
                    if not isinstance(fields, dict) or not isinstance(answers, dict) or not isinstance(documents, list):
                        errors.append("Security Gate: draft payload structure is invalid.")
                    else:
                        actual_hash = self._draft_payload_hash(fields, answers, documents)
                        if draft[4] != actual_hash:
                            errors.append("Security Gate: immutable draft payload hash mismatch.")
                except Exception as exc:
                    errors.append(f"Security Gate: draft JSON integrity check failed: {exc}")

            if app[3] is not None and int(app[3]) == int(version) and app[4]:
                # Direct gate call is not the idempotency path. submit_application handles this case before the gate.
                errors.append("Security Gate: application is already submitted for this version; use idempotent replay path.")
        finally:
            conn.close()
        return len(errors) == 0, errors

    async def _execute_browser_submission(
        self,
        application_id: str,
        version: int,
        force_failure_scenario: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Idempotency must be checked before the policy gate and before any browser call.
        conn = self._get_db_connection()
        try:
            row = conn.execute(
                "SELECT submitted_version, confirmation_id, current_stage FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
        finally:
            conn.close()

        if row and row[0] == version and row[1]:
            self.log_audit_event(application_id, "IDEMPOTENT_REPLAY_SKIPPED", {"version": version, "confirmation_id": row[1]})
            return {
                "success": True,
                "status": "ALREADY_SUBMITTED",
                "application_id": application_id,
                "submitted_version": version,
                "confirmation_id": row[1],
            }

        safe, errors = self.pre_submission_policy_gate(application_id, version)
        if not safe:
            self.log_audit_event(application_id, "SUBMISSION_HALTED_POLICY_GATE", {"errors": errors})
            return BrowserExecutionFailure(success=False, error_type="POLICY_GATE_REJECTION", error=" | ".join(errors), execution_mode="DEMO" if DEMO_MODE else "REAL").to_dict()

        ok, message = self.fsm.transition(application_id, ApplicationStatus.SUBMITTING, reason=f"Submitting Draft V{version}")
        if not ok:
            return BrowserExecutionFailure(success=False, error_type="INVALID_STATE", error=message, execution_mode="DEMO" if DEMO_MODE else "REAL").to_dict()

        # Resolve approved documents after state transition.
        conn = self._get_db_connection()
        try:
            draft = conn.execute("SELECT documents FROM application_drafts WHERE application_id = ? AND version = ?", (application_id, version)).fetchone()
        finally:
            conn.close()

        documents = json.loads(draft[0] or "[]") if draft else []
        resume_path = None
        if documents and isinstance(documents[0], dict):
            resume_path = documents[0].get("path")

        self.log_audit_event(application_id, "PLAYWRIGHT_STARTED", {"version": version, "resume_path": resume_path}, run_id="SUBMISSION")

        if force_failure_scenario == "BLOCKED_BY_CAPTCHA":
            self.fail_application(application_id, "CAPTCHA encountered. Manual intervention required.", "CAPTCHA_REQUIRED")
            self.log_audit_event(application_id, "CAPTCHA_PAUSE_TRIGGERED", {})
            return BrowserExecutionFailure(success=False, error_type="BLOCKED_BY_CAPTCHA", error="Bot challenge detected.", execution_mode="DEMO" if DEMO_MODE else "REAL").to_dict()

        if force_failure_scenario == "FIELD_NOT_FOUND":
            self.fail_application(application_id, "Required browser field was not found.", "FIELD_NOT_FOUND")
            return BrowserExecutionFailure(success=False, error_type="FIELD_NOT_FOUND", error="Required form field could not be mapped.", execution_mode="DEMO" if DEMO_MODE else "REAL").to_dict()

        confirmation_raw = ""
        execution_mode = "DEMO" if DEMO_MODE else "REAL"

        try:
            if DEMO_MODE:
                confirmation_raw = f"CONF-DEMO-{uuid.uuid4().hex[:10].upper()}"
            else:
                from playwright.async_api import async_playwright

                portal_path = os.path.abspath(os.getenv("APPLYX_PORTAL_PATH", "mock_job_portal.html"))
                if not os.path.exists(portal_path):
                    raise FileNotFoundError(f"Portal fixture not found: {portal_path}")
                async with async_playwright() as playwright:
                    browser = await playwright.chromium.launch(headless=True)
                    try:
                        page = await browser.new_page()
                        await page.goto(f"file:///{portal_path.replace(os.sep, '/')}")
                        await page.wait_for_selector("#submitBtn", timeout=5000)

                        if resume_path and os.path.exists(resume_path):
                            file_input = await page.query_selector("#resumeUpload")
                            if file_input:
                                await file_input.set_input_files(resume_path)

                        await page.click("#submitBtn")
                        await page.wait_for_timeout(250)
                        conf = await page.locator("#conf").inner_text()
                        confirmation_raw = str(conf).strip()
                    finally:
                        await browser.close()

            if not re.fullmatch(r"^CONF-[A-Z0-9-]+$", confirmation_raw):
                raise ValueError(f"CONFIRMATION_FORMAT_INVALID: '{confirmation_raw}'")
            confirmation_id = f"{confirmation_raw}-APP{application_id[-6:].upper()}-V{version}"
            if not re.fullmatch(r"^CONF-[A-Z0-9-]+$", confirmation_id):
                raise ValueError(f"FINAL_CONFIRMATION_FORMAT_INVALID: '{confirmation_id}'")
        except Exception as exc:
            self.fail_application(application_id, f"Browser automation failed: {exc}", "BROWSER_EXECUTION_FAILED")
            return BrowserExecutionFailure(success=False, error_type="BROWSER_EXECUTION_FAILED", error=f"Browser automation failed: {exc}", execution_mode=execution_mode).to_dict()

        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Verify the submission transaction still targets the approved version.
            row = conn.execute("SELECT current_stage, approved_version, approved_draft_id FROM applications WHERE id = ?", (application_id,)).fetchone()
            if not row or row[0] != ApplicationStatus.SUBMITTING.value or row[1] != version:
                conn.rollback()
                raise RuntimeError("Submission state changed unexpectedly before commit.")

            draft = conn.execute("SELECT id FROM application_drafts WHERE application_id = ? AND version = ?", (application_id, version)).fetchone()
            if not draft or draft[0] != row[2]:
                conn.rollback()
                raise RuntimeError("Approved draft identity changed before submission commit.")

            now_iso = datetime.now().isoformat()
            conn.execute(
                """
                UPDATE applications
                SET submitted_version = ?, confirmation_id = ?, submitted_at = ?, error_message = NULL, updated_at = ?
                WHERE id = ?
                """,
                (version, confirmation_id, now_iso, now_iso, application_id),
            )
            ok, message = self.fsm.transition(
                application_id,
                ApplicationStatus.SUBMITTED,
                reason=f"Submission verified: {confirmation_id}",
                external_conn=conn,
            )
            if not ok:
                conn.rollback()
                raise RuntimeError(message)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            self.fail_application(application_id, f"Submission transaction failed: {exc}", "TRANSACTION_ERROR")
            return BrowserExecutionFailure(success=False, error_type="BROWSER_EXECUTION_FAILED", error=str(exc), execution_mode=execution_mode).to_dict()
        finally:
            conn.close()

        self.log_audit_event(application_id, "APPLICATION_SUBMITTED", {"confirmation_id": confirmation_id, "version": version, "mode": execution_mode})
        return BrowserExecutionSuccess(application_id=application_id, draft_version=version, confirmation_id=confirmation_id, execution_mode=execution_mode).to_dict()

    async def process_opportunity(
        self,
        user_id: str,
        raw_opportunity: Any,
        agent_mode: Optional[str] = None,
    ) -> OrchestratorResult:
        """Run the frozen intake pipeline and delegate reasoning to Strands."""
        logs: List[str] = []
        try:
            opportunity = self.normalize_opportunity(raw_opportunity)
        except Exception as exc:
            application_id = self.generate_safe_application_id()
            return OrchestratorResult(application_id, "UNKNOWN", ApplicationStatus.FAILED.value, "FAILED", reason=f"Opportunity normalization failed: {exc}")

        opportunity_id = opportunity.id
        logs.append(f"[{datetime.now().isoformat()}] Intake Triggered: {opportunity_id} ({opportunity.title})")

        existing = self.check_duplicate(user_id, opportunity_id)
        if existing:
            self.log_audit_event(existing["id"], "DUPLICATE_INTAKE_BLOCKED", {"attempted_opportunity": opportunity_id})
            return OrchestratorResult(
                application_id=existing["id"],
                opportunity_id=opportunity_id,
                stage=existing["current_stage"] or existing["status"],
                decision="DUPLICATE",
                confidence=1.0,
                risk="NONE",
                reason=f"Duplicate intake blocked; existing application '{existing['id']}' preserved.",
                logs=logs,
            )

        application_id = self.generate_safe_application_id()
        now_iso = datetime.now().isoformat()
        conn = self._get_db_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO applications (
                    id, user_id, opportunity_id, current_stage, status,
                    approved_version, approved_draft_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (application_id, user_id, opportunity_id, ApplicationStatus.DISCOVERED.value, ApplicationStatus.DISCOVERED.value, now_iso, now_iso),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.rollback()
            existing = self.check_duplicate(user_id, opportunity_id)
            if existing:
                self.log_audit_event(existing["id"], "DUPLICATE_INTEGRITY_CONSTRAINED", {"attempted_opportunity": opportunity_id})
                return OrchestratorResult(existing["id"], opportunity_id, existing["current_stage"] or existing["status"], "DUPLICATE", confidence=1.0, risk="NONE", reason="Database UNIQUE constraint blocked duplicate intake.", logs=logs)
            raise
        except Exception as exc:
            conn.rollback()
            return OrchestratorResult(application_id, opportunity_id, ApplicationStatus.FAILED.value, "FAILED", reason=f"Application record creation failed: {exc}", logs=logs)
        finally:
            conn.close()

        try:
            for status, reason in [
                (ApplicationStatus.NORMALIZED, "Opportunity normalized"),
                (ApplicationStatus.DUPLICATE_CHECKED, "Verified application uniqueness"),
                (ApplicationStatus.ANALYZED, "Ready for Strands workflow"),
            ]:
                ok, message = self.fsm.transition(application_id, status, reason=reason, metadata={"run_id": "RUN-1"})
                if not ok:
                    raise RuntimeError(message)
        except Exception as exc:
            self.fail_application(application_id, f"Lifecycle transition failed: {exc}", "LIFECYCLE_TRANSITION_ERROR", run_id="RUN-1")
            return OrchestratorResult(application_id, opportunity_id, ApplicationStatus.FAILED.value, "FAILED", reason=str(exc), logs=logs)

        from core.strands_agent import run_strands_application_flow
        try:
            result = await run_strands_application_flow(
                user_id=user_id,
                raw_opportunity=opportunity.to_dict(),
                application_id=application_id,
                orchestrator=self,
                run_id="RUN-1",
                db_path=self.db_path,
                agent_mode=agent_mode,
            )
        except Exception as exc:
            self.fail_application(application_id, f"Strands execution failed: {exc}", "AGENT_EXECUTION_ERROR", run_id="RUN-1")
            return OrchestratorResult(application_id, opportunity_id, ApplicationStatus.FAILED.value, "FAILED", reason=str(exc), logs=logs)

        status = result.get("status")
        action = result.get("decision")
        traces = result.get("traces", [])
        if status != "COMPLETED" or action not in {"APPLY", "REVIEW", "SKIP"}:
            reason = result.get("reason") or "INCOMPLETE_WORKFLOW"
            self.fail_application(application_id, reason, result.get("terminal_outcome", "INCOMPLETE_WORKFLOW"), run_id="RUN-1")
            return OrchestratorResult(application_id, opportunity_id, ApplicationStatus.FAILED.value, "FAILED", confidence=result.get("confidence", 0.0), fit_score=result.get("fit_score", 0), risk=result.get("risk", "UNKNOWN"), reason=reason, logs=logs, match_analysis=result.get("match_analysis"), traces=traces)

        base_kwargs = dict(
            application_id=application_id,
            opportunity_id=opportunity_id,
            confidence=float(result.get("confidence", 0.85)),
            fit_score=int(result.get("fit_score", 0)),
            evidence_coverage=float(result.get("evidence_coverage", 0.0)),
            risk=result.get("risk", "UNKNOWN"),
            reason=result.get("reason"),
            logs=logs,
            match_analysis=result.get("match_analysis"),
            traces=traces,
        )

        if action == "REVIEW":
            ok1, msg1 = self.fsm.transition(application_id, ApplicationStatus.REVIEW_REQUIRED, reason=result.get("reason"), metadata={"run_id": "RUN-1"})
            ok2, msg2 = self.fsm.transition(application_id, ApplicationStatus.CLARIFICATION_REQUIRED, reason="Candidate clarification required", metadata={"run_id": "RUN-1", "prompt": result.get("clarification_prompt")})
            if not (ok1 and ok2):
                self.fail_application(application_id, f"Review transition failed: {msg1 if not ok1 else msg2}", "FSM_TRANSITION_ERROR", run_id="RUN-1")
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED")
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.CLARIFICATION_REQUIRED.value, decision="REVIEW", clarification_prompt=result.get("clarification_prompt"), clarification_id=result.get("clarification_id"), clarification_attempts=0)

        if action == "SKIP":
            ok, message = self.fsm.transition(application_id, ApplicationStatus.SKIPPED, reason=result.get("reason"), metadata={"run_id": "RUN-1"})
            if not ok:
                self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-1")
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED")
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.SKIPPED.value, decision="SKIP")

        draft = self.get_draft(application_id, 1)
        if not draft:
            reason = "Draft persistence missing: APPLY was returned without persisted Draft V1."
            self.fail_application(application_id, reason, "DRAFT_PERSISTENCE_MISSING", run_id="RUN-1")
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=reason)

        for status_state, reason_text in [
            (ApplicationStatus.APPLY, "Eligibility confirmed by authoritative decision"),
            (ApplicationStatus.DRAFT_CREATED, f"Draft generated ({draft['id']})"),
            (ApplicationStatus.AWAITING_APPROVAL, "Locked awaiting human approval"),
        ]:
            ok, message = self.fsm.transition(application_id, status_state, reason=reason_text, metadata={"run_id": "RUN-1"})
            if not ok:
                self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-1")
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=message)

        return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.AWAITING_APPROVAL.value, decision="APPLY", draft_version=1)

    def _latest_clarification_id(self, application_id: str) -> Optional[str]:
        conn = self._get_db_connection()
        try:
            row = conn.execute(
                "SELECT event_payload FROM application_events WHERE application_id = ? AND event_type = 'CLARIFICATION_REQUESTED' ORDER BY id DESC LIMIT 1",
                (application_id,),
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0]).get("clarification_id")
            except Exception:
                return None
        finally:
            conn.close()

    async def submit_clarification_and_reevaluate(
        self,
        application_id: str,
        user_id: str,
        raw_opportunity: Any,
        clarification_key: str,
        clarification_answer: Any,
        current_attempts: Optional[int] = None,
        clarification_id: Optional[str] = None,
        agent_mode: Optional[str] = None,
    ) -> OrchestratorResult:
        """Apply exactly one user clarification, then launch Run #2 on the same application."""
        conn = self._get_db_connection()
        try:
            app_row = conn.execute(
                "SELECT user_id, opportunity_id, current_stage, clarification_attempts, clarification_id FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
        finally:
            conn.close()
        if not app_row:
            return OrchestratorResult(application_id, "UNKNOWN", ApplicationStatus.FAILED.value, "FAILED", reason="Application not found.")
        if app_row[0] != user_id:
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.FAILED.value, "FAILED", reason="User/application ownership mismatch.")
        if app_row[2] != ApplicationStatus.CLARIFICATION_REQUIRED.value:
            return OrchestratorResult(application_id, app_row[1], app_row[2], "REVIEW", reason=f"Clarification is not allowed from state {app_row[2]}.")

        trusted_attempts = int(app_row[3] or 0)
        # UI/API supplied current_attempts is informational only; SQLite is authoritative.
        del current_attempts
        if app_row[4] and clarification_id and app_row[4] != clarification_id:
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.CLARIFICATION_REQUIRED.value, "REVIEW", reason="Clarification ID does not match the active application clarification.", clarification_id=app_row[4], clarification_attempts=trusted_attempts)
        if trusted_attempts >= MAX_CLARIFICATION_ATTEMPTS:
            reason = "MAX_CLARIFICATION_ATTEMPTS_EXCEEDED"
            ok, message = self.fsm.transition(application_id, ApplicationStatus.SKIPPED, reason=reason, metadata={"run_id": "RUN-2"})
            if not ok:
                self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-2")
                return OrchestratorResult(application_id, app_row[1], ApplicationStatus.FAILED.value, "FAILED", reason=message)
            self.log_audit_event(application_id, "CLARIFICATION_LIMIT_EXCEEDED", {"attempts": trusted_attempts}, run_id="RUN-2")
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.SKIPPED.value, "SKIP", reason=reason, clarification_attempts=trusted_attempts, traces=[])

        if clarification_key not in ALLOWED_KB_FIELDS:
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.CLARIFICATION_REQUIRED.value, "REVIEW", reason=f"Unsupported Knowledge Base field: {clarification_key}")

        clr_id = clarification_id or f"CLR-{uuid.uuid4()}"
        if not re.fullmatch(r"CLR-[0-9a-fA-F-]{36}", clr_id):
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.CLARIFICATION_REQUIRED.value, "REVIEW", reason="Invalid clarification ID format.")

        # The single KB mutation is the only mutation before Run #2.
        try:
            self.update_user_kb(user_id, clarification_key, clarification_answer, application_id=application_id, clarification_id=clr_id)
        except Exception as exc:
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.CLARIFICATION_REQUIRED.value, "REVIEW", reason=f"Clarification update failed: {exc}", clarification_id=clr_id, clarification_attempts=trusted_attempts)

        next_attempt = trusted_attempts + 1
        ok, message = self.fsm.transition(application_id, ApplicationStatus.RE_EVALUATING, reason="Clarification accepted; starting Run #2", metadata={"run_id": "RUN-2", "clarification_id": clr_id})
        if not ok:
            self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-2")
            return OrchestratorResult(application_id, app_row[1], ApplicationStatus.FAILED.value, "FAILED", reason=message, clarification_id=clr_id, clarification_attempts=next_attempt)

        from core.strands_agent import run_strands_application_flow
        opportunity = self.normalize_opportunity(raw_opportunity)
        try:
            result = await run_strands_application_flow(
                user_id=user_id,
                raw_opportunity=opportunity.to_dict(),
                application_id=application_id,
                orchestrator=self,
                run_id="RUN-2",
                db_path=self.db_path,
                clarification_id=clr_id,
                clarification_key=clarification_key,
                clarification_attempt=next_attempt,
                agent_mode=agent_mode,
            )
        except Exception as exc:
            self.fail_application(application_id, f"Agent Run #2 failed: {exc}", "RUN2_EXECUTION_ERROR", run_id="RUN-2")
            return OrchestratorResult(application_id, opportunity.id, ApplicationStatus.FAILED.value, "FAILED", reason=str(exc), clarification_id=clr_id, clarification_attempts=next_attempt, traces=[])

        traces = result.get("traces", [])
        action = result.get("decision")
        base_kwargs = dict(
            application_id=application_id,
            opportunity_id=opportunity.id,
            confidence=float(result.get("confidence", 0.85)),
            fit_score=int(result.get("fit_score", 0)),
            evidence_coverage=float(result.get("evidence_coverage", 0.0)),
            risk=result.get("risk", "UNKNOWN"),
            reason=result.get("reason"),
            clarification_id=clr_id,
            clarification_attempts=next_attempt,
            match_analysis=result.get("match_analysis"),
            traces=traces,
        )

        if result.get("status") != "COMPLETED" or action not in {"APPLY", "REVIEW", "SKIP"}:
            reason = result.get("reason") or "INCOMPLETE_WORKFLOW"
            self.fail_application(application_id, reason, result.get("terminal_outcome", "INCOMPLETE_WORKFLOW"), run_id="RUN-2")
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=reason)

        if action == "REVIEW":
            if next_attempt >= MAX_CLARIFICATION_ATTEMPTS:
                reason = "MAX_CLARIFICATION_ATTEMPTS_EXCEEDED"
                ok, message = self.fsm.transition(application_id, ApplicationStatus.SKIPPED, reason=reason, metadata={"run_id": "RUN-2"})
                if not ok:
                    self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-2")
                    return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=message)
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.SKIPPED.value, decision="SKIP", reason=reason)

            self.fsm.transition(application_id, ApplicationStatus.REVIEW_REQUIRED, reason=result.get("reason"), metadata={"run_id": "RUN-2"})
            self.fsm.transition(application_id, ApplicationStatus.CLARIFICATION_REQUIRED, reason=f"Attempt {next_attempt} unresolved", metadata={"run_id": "RUN-2"})
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.CLARIFICATION_REQUIRED.value, decision="REVIEW", clarification_prompt=result.get("clarification_prompt"))

        if action == "SKIP":
            ok, message = self.fsm.transition(application_id, ApplicationStatus.SKIPPED, reason=result.get("reason"), metadata={"run_id": "RUN-2"})
            if not ok:
                self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-2")
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=message)
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.SKIPPED.value, decision="SKIP")

        draft = self.get_draft(application_id, 1)
        if not draft:
            reason = "Draft persistence missing: Run #2 returned APPLY without persisted Draft V1."
            self.fail_application(application_id, reason, "DRAFT_PERSISTENCE_MISSING", run_id="RUN-2")
            return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=reason)

        for state, reason_text in [
            (ApplicationStatus.APPLY, "Eligibility confirmed after clarification"),
            (ApplicationStatus.DRAFT_CREATED, f"Draft generated ({draft['id']})"),
            (ApplicationStatus.AWAITING_APPROVAL, "Locked awaiting human approval"),
        ]:
            ok, message = self.fsm.transition(application_id, state, reason=reason_text, metadata={"run_id": "RUN-2"})
            if not ok:
                self.fail_application(application_id, message, "FSM_TRANSITION_ERROR", run_id="RUN-2")
                return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.FAILED.value, decision="FAILED", reason=message)

        return OrchestratorResult(**base_kwargs, stage=ApplicationStatus.AWAITING_APPROVAL.value, decision="APPLY", draft_version=1)
