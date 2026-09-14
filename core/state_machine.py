"""
core/state_machine.py
Authoritative lifecycle state machine for ApplyX.

The orchestrator owns business lifecycle mutations; this module only validates
and persists legal state transitions. All transition/audit writes are atomic
when the caller supplies an existing SQLite transaction.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ApplicationStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    NORMALIZED = "NORMALIZED"
    DUPLICATE_CHECKED = "DUPLICATE_CHECKED"
    ANALYZED = "ANALYZED"
    APPLY = "APPLY"
    DRAFT_CREATED = "DRAFT_CREATED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"

    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    RE_EVALUATING = "RE_EVALUATING"
    DUPLICATE = "DUPLICATE"
    TERMINATED = "TERMINATED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: Dict[ApplicationStatus, List[ApplicationStatus]] = {
    ApplicationStatus.DISCOVERED: [ApplicationStatus.NORMALIZED, ApplicationStatus.FAILED],
    ApplicationStatus.NORMALIZED: [ApplicationStatus.DUPLICATE_CHECKED, ApplicationStatus.FAILED],
    ApplicationStatus.DUPLICATE_CHECKED: [ApplicationStatus.ANALYZED, ApplicationStatus.DUPLICATE, ApplicationStatus.FAILED],
    ApplicationStatus.DUPLICATE: [ApplicationStatus.TERMINATED],
    ApplicationStatus.ANALYZED: [ApplicationStatus.APPLY, ApplicationStatus.REVIEW_REQUIRED, ApplicationStatus.SKIPPED, ApplicationStatus.FAILED],
    ApplicationStatus.REVIEW_REQUIRED: [ApplicationStatus.CLARIFICATION_REQUIRED, ApplicationStatus.SKIPPED, ApplicationStatus.FAILED],
    ApplicationStatus.CLARIFICATION_REQUIRED: [ApplicationStatus.RE_EVALUATING, ApplicationStatus.SKIPPED, ApplicationStatus.FAILED],
    ApplicationStatus.RE_EVALUATING: [ApplicationStatus.APPLY, ApplicationStatus.REVIEW_REQUIRED, ApplicationStatus.SKIPPED, ApplicationStatus.FAILED],
    ApplicationStatus.APPLY: [ApplicationStatus.DRAFT_CREATED, ApplicationStatus.FAILED],
    ApplicationStatus.DRAFT_CREATED: [ApplicationStatus.AWAITING_APPROVAL, ApplicationStatus.FAILED],
    ApplicationStatus.AWAITING_APPROVAL: [ApplicationStatus.APPROVED, ApplicationStatus.RE_EVALUATING, ApplicationStatus.SKIPPED, ApplicationStatus.FAILED],
    ApplicationStatus.APPROVED: [ApplicationStatus.SUBMITTING, ApplicationStatus.AWAITING_APPROVAL, ApplicationStatus.FAILED],
    ApplicationStatus.SUBMITTING: [ApplicationStatus.SUBMITTED, ApplicationStatus.FAILED],
    ApplicationStatus.SUBMITTED: [],
    ApplicationStatus.TERMINATED: [],
    ApplicationStatus.SKIPPED: [ApplicationStatus.RE_EVALUATING],
    ApplicationStatus.FAILED: [ApplicationStatus.AWAITING_APPROVAL, ApplicationStatus.REVIEW_REQUIRED, ApplicationStatus.RE_EVALUATING],
}


class AuthoritativeStateMachine:
    def __init__(self, db_path: str = "agent_applications.db") -> None:
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=60.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass
        return conn

    def get_current_status(
        self,
        application_id: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> Optional[ApplicationStatus]:
        close_needed = conn is None
        conn = conn or self._get_connection()
        try:
            row = conn.execute(
                "SELECT current_stage, status FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
            if not row:
                return None

            raw = row["current_stage"] or row["status"]
            if not raw:
                return None
            normalized = str(raw).strip().upper().replace(" ", "_")
            try:
                return ApplicationStatus(normalized)
            except ValueError:
                logger.error("Unknown application state '%s' for %s", raw, application_id)
                return None
        finally:
            if close_needed:
                conn.close()

    @staticmethod
    def _next_sequence(conn: sqlite3.Connection, application_id: str, run_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM application_events WHERE application_id = ? AND run_id = ?",
            (application_id, run_id),
        ).fetchone()
        return int(row[0]) if row else 1

    def transition(
        self,
        application_id: str,
        target_status: ApplicationStatus,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        external_conn: Optional[sqlite3.Connection] = None,
    ) -> Tuple[bool, str]:
        """Persist one legal lifecycle transition atomically."""
        close_needed = external_conn is None
        conn = external_conn or self._get_connection()

        try:
            if close_needed:
                conn.execute("BEGIN IMMEDIATE")

            current_status = self.get_current_status(application_id, conn)
            if current_status is None:
                if close_needed:
                    conn.rollback()
                return False, f"Application '{application_id}' not found or has an invalid state."

            metadata = metadata or {}

            # Idempotent same-state transition is intentionally a no-op.
            if current_status == target_status:
                if close_needed:
                    conn.commit()
                return True, f"Application is already in state '{target_status.value}'."

            allowed = ALLOWED_TRANSITIONS.get(current_status, [])
            if target_status not in allowed:
                permitted = [state.value for state in allowed] or ["None (Terminal)"]
                message = (
                    f"Illegal State Transition: Cannot move '{application_id}' from "
                    f"{current_status.value} -> {target_status.value}. Permitted: {permitted}"
                )
                logger.warning(message)
                if close_needed:
                    conn.rollback()
                return False, message

            now_iso = datetime.now().isoformat()
            conn.execute(
                """
                UPDATE applications
                SET current_stage = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (target_status.value, target_status.value, now_iso, application_id),
            )

            run_id = str(metadata.get("run_id") or "RUN-1")
            sequence_number = self._next_sequence(conn, application_id, run_id)
            payload = {
                "from_state": current_status.value,
                "to_state": target_status.value,
                "reason": reason or "State Transition",
                "metadata": metadata,
            }

            conn.execute(
                """
                INSERT INTO application_events (
                    application_id, run_id, sequence_number, event_type,
                    stage, tool_name, event_payload, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    run_id,
                    sequence_number,
                    f"FSM_{target_status.value}",
                    target_status.value,
                    None,
                    json.dumps(payload, default=str),
                    now_iso,
                ),
            )

            if close_needed:
                conn.commit()
            return True, f"Transitioned {current_status.value} -> {target_status.value}"
        except Exception as exc:
            if close_needed:
                conn.rollback()
            logger.exception("Transition failure for %s", application_id)
            return False, f"Database transaction failed: {exc}"
        finally:
            if close_needed:
                conn.close()
