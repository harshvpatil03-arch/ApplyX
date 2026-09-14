"""
core/strands_agent.py
Real Strands Agents integration plus an explicit mock execution backend.

No real->mock fallback exists. REAL mode requires the installed Strands SDK;
MOCK mode is selected explicitly by caller/environment.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import os
import queue
import sqlite3
import threading
from typing import Any, Dict, List, Optional

try:
    from strands import Agent, tool as strands_tool
    from strands.types.agent import Limits
    from strands.hooks import BeforeInvocationEvent, AfterInvocationEvent, BeforeToolCallEvent, AfterToolCallEvent
    STRANDS_AVAILABLE = True
except ImportError:
    Agent = None  # type: ignore[assignment]
    strands_tool = None  # type: ignore[assignment]
    Limits = None  # type: ignore[assignment]
    BeforeInvocationEvent = AfterInvocationEvent = BeforeToolCallEvent = AfterToolCallEvent = object  # type: ignore[assignment]
    STRANDS_AVAILABLE = False

from core.agent_tools import (
    analyze_candidate_match,
    generate_application_draft,
    get_candidate_profile,
    make_application_decision,
    normalize_opportunity,
    request_clarification,
    re_evaluate_application,
    get_orchestrator,
    set_orchestrator,
    reset_orchestrator,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_CALL_BUDGET = 15
_MAX_TURN_COUNT = 12

HARDCODED_GEMINI_API_KEY = "PUT_UR_API_KEY_HERE"

APPLYX_SYSTEM_PROMPT = """
You are ApplyX, an application workflow agent.

You MUST execute the available tools in this exact order:
Run 1: get_candidate_profile -> normalize_opportunity -> analyze_candidate_match -> make_application_decision -> branch
Run 2: get_candidate_profile -> normalize_opportunity -> re_evaluate_application -> make_application_decision -> branch

Branch rules:
- APPLY: call generate_application_draft exactly once, then stop.
- REVIEW: call request_clarification exactly once, then stop.
- SKIP: stop without draft or submission.

Rules:
- Structured tool results, especially make_application_decision, are authoritative.
- Never invent or alter candidate facts.
- Never submit applications during Run 1 or Run 2.
- Never call a tool twice.
- Infrastructure IDs are supplied by trusted invocation state; do not invent them.
""".strip()


@dataclass
class AgentInvocationContext:
    run_id: str
    application_id: str
    user_id: str
    db_path: str
    opportunity_payload: Dict[str, Any]
    clarification_id: Optional[str] = None
    clarification_key: Optional[str] = None
    clarification_attempt: int = 0
    tool_call_count: int = 0
    executed_tools: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    decision_result: Optional[Dict[str, Any]] = None
    cancellation_reason: Optional[str] = None
    terminal_agent_result: Optional[Any] = None
    tool_error: Optional[str] = None


_context_invocation: contextvars.ContextVar[Optional[AgentInvocationContext]] = contextvars.ContextVar(
    "applyx_agent_invocation", default=None
)


class TraceSink:
    """Durable SQLite trace source of truth plus isolated live queues."""

    _instance: Optional["TraceSink"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[tuple[queue.Queue, Optional[str]]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TraceSink":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_live_queue(self, application_id: str = "*", user_id: Optional[str] = None) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(application_id, []).append((q, user_id))
        return q

    def unregister_live_queue(self, application_id: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(application_id, [])
            self._subscribers[application_id] = [(existing, uid) for existing, uid in subscribers if existing is not q]
            if not self._subscribers[application_id]:
                self._subscribers.pop(application_id, None)

    def drain_live_queue(self, q: queue.Queue) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        while True:
            try:
                output.append(q.get_nowait())
            except queue.Empty:
                return output

    def _persist(self, trace_record: Dict[str, Any], db_path: str) -> Dict[str, Any]:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            app_id = trace_record["application_id"]
            run_id = trace_record["run_id"]
            seq = trace_record.get("sequence_number")
            if seq is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM application_events WHERE application_id = ? AND run_id = ?",
                    (app_id, run_id),
                ).fetchone()
                seq = int(row[0]) if row else 1
                trace_record["sequence_number"] = seq
            now_iso = trace_record.get("timestamp") or datetime.now().isoformat()
            trace_record["timestamp"] = now_iso
            stage = trace_record.get("event_type", "TRACE_EVENT")
            conn.execute(
                """
                INSERT INTO application_events (
                    application_id, run_id, sequence_number, event_type,
                    stage, tool_name, event_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    run_id,
                    int(seq),
                    f"STRANDS_{stage}",
                    f"STRANDS_{stage}",
                    trace_record.get("tool_name"),
                    json.dumps(trace_record, default=str),
                    now_iso,
                ),
            )
            conn.commit()
            return trace_record
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def emit(self, trace_record: Dict[str, Any], db_path: str) -> Dict[str, Any]:
        """Persist first, then publish to live subscribers."""
        persisted = self._persist(trace_record, db_path)
        app_id = persisted.get("application_id", "*")
        user_id = persisted.get("user_id")
        with self._lock:
            targets = list(self._subscribers.get(app_id, [])) + list(self._subscribers.get("*", []))
        for q, filter_user in targets:
            if filter_user is None or filter_user == user_id:
                q.put_nowait(dict(persisted))
        return persisted


def record_agent_trace_event(
    event_type: str,
    details: Dict[str, Any],
    invocation: Optional[AgentInvocationContext] = None,
    tool_name: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = invocation or _context_invocation.get()
    if ctx is None:
        raise RuntimeError("Cannot emit agent trace without invocation context.")

    safe_details: Dict[str, Any] = {}
    for key, value in (details or {}).items():
        try:
            json.dumps(value)
            safe_details[key] = value
        except (TypeError, ValueError):
            safe_details[key] = str(value)[:500]

    record = {
        "application_id": ctx.application_id,
        "run_id": ctx.run_id,
        "sequence_number": None,
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "tool_name": tool_name,
        "user_id": ctx.user_id,
        "details": safe_details,
    }
    persisted = TraceSink.get_instance().emit(record, ctx.db_path)
    _trace_memory.setdefault((ctx.application_id, ctx.run_id), []).append(dict(persisted))
    return persisted


_trace_memory: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
_trace_memory_lock = threading.Lock()


def get_invocation_traces(application_id: str, run_id: str) -> List[Dict[str, Any]]:
    with _trace_memory_lock:
        return list(_trace_memory.get((application_id, run_id), []))


def clear_invocation_traces(application_id: str, run_id: str) -> None:
    with _trace_memory_lock:
        _trace_memory.pop((application_id, run_id), None)


def clear_active_agent_traces() -> None:
    with _trace_memory_lock:
        _trace_memory.clear()


def extract_tool_result(event: Any) -> Optional[Dict[str, Any]]:
    if event is None:
        return None
    candidates = [event.get("result") if isinstance(event, dict) else None, event]
    if not isinstance(event, dict):
        candidates += [getattr(event, "result", None), getattr(event, "output", None), getattr(event, "tool_result", None)]
    for candidate in candidates:
        if isinstance(candidate, dict):
            if isinstance(candidate.get("result"), dict):
                return candidate["result"]
            content = candidate.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    nested = item.get("json") or item.get("data") or item.get("value")
                    if isinstance(nested, dict):
                        return nested
                    text = item.get("text")
                    if isinstance(text, str):
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                return parsed
                        except (TypeError, ValueError):
                            pass
            return candidate
        if hasattr(candidate, "to_dict") and callable(candidate.to_dict):
            try:
                converted = candidate.to_dict()
                if isinstance(converted, dict):
                    return converted
            except Exception:
                pass
    return None


def extract_terminal_result(event: Any) -> Optional[Any]:
    if event is None:
        return None
    if isinstance(event, dict) and "stop_reason" in event:
        return event
    if getattr(event, "stop_reason", None) is not None:
        return event
    for name in ("result", "agent_result", "terminal_result"):
        value = getattr(event, name, None)
        if value is not None and (getattr(value, "stop_reason", None) is not None or isinstance(value, dict) and "stop_reason" in value):
            return value
    return None


def _event_tool_name(event: Any) -> str:
    selected = getattr(event, "selected_tool", None)
    if isinstance(selected, dict):
        selected_name = selected.get("name")
    else:
        selected_name = getattr(selected, "name", None)

    tool_use = getattr(event, "tool_use", None)
    tool_use_name = tool_use.get("name") if isinstance(tool_use, dict) else None

    # Current Strands uses event.tool_use["name"]; older adapters may expose
    # selected_tool/name. Keep both forms for compatibility.
    return str(
        tool_use_name
        or selected_name
        or getattr(event, "tool_name", None)
        or "unknown_tool"
    )


def on_before_invocation(event: Any) -> None:
    ctx = _context_invocation.get()
    if ctx:
        record_agent_trace_event("STRANDS_AGENT_STARTED", {"message": "Invocation started"}, invocation=ctx)


def on_after_invocation(event: Any) -> None:
    ctx = _context_invocation.get()
    if ctx:
        terminal = extract_terminal_result(event)
        if terminal is not None:
            ctx.terminal_agent_result = terminal
        record_agent_trace_event("STRANDS_AGENT_COMPLETED", {"message": "Invocation completed"}, invocation=ctx)


def on_before_tool_call(event: Any) -> None:
    ctx = _context_invocation.get()
    if not ctx:
        return
    ctx.tool_call_count += 1
    tool_name = _event_tool_name(event)

    # The frozen workflow requires every tool to execute at most once.
    if tool_name in ctx.executed_tools:
        ctx.cancellation_reason = "WORKFLOW_VIOLATION"
        try:
            setattr(event, "cancel_tool", f"Tool '{tool_name}' was already executed; duplicate calls are forbidden.")
        except Exception:
            pass
        record_agent_trace_event(
            "DUPLICATE_TOOL_CALL_BLOCKED",
            {"tool_call_count": ctx.tool_call_count},
            invocation=ctx,
            tool_name=tool_name,
        )
        return

    if ctx.tool_call_count > _MAX_TOOL_CALL_BUDGET:
        ctx.cancellation_reason = "TOOL_LIMIT"
        try:
            setattr(event, "cancel_tool", True)
        except Exception:
            pass
        record_agent_trace_event(
            "AGENT_TOOL_BUDGET_EXCEEDED",
            {"count": ctx.tool_call_count, "max": _MAX_TOOL_CALL_BUDGET},
            invocation=ctx,
            tool_name=tool_name,
        )
        return
    record_agent_trace_event("BEFORE_TOOL_CALL", {"tool_call_count": ctx.tool_call_count}, invocation=ctx, tool_name=tool_name)


def on_after_tool_call(event: Any) -> None:
    ctx = _context_invocation.get()
    if not ctx:
        return
    tool_name = _event_tool_name(event)
    result = getattr(event, "result", None)
    exception = getattr(event, "exception", None)
    if exception:
        ctx.tool_error = str(exception)
    result_dict = extract_tool_result(event) if exception is None else None
    if exception is None:
        ctx.executed_tools.append(tool_name)
        if isinstance(result_dict, dict):
            ctx.tool_results[tool_name] = result_dict
            if tool_name == "make_application_decision":
                ctx.decision_result = result_dict
            if tool_name == "request_clarification" and result_dict.get("clarification_id"):
                ctx.clarification_id = str(result_dict["clarification_id"])
    record_agent_trace_event(
        "AFTER_TOOL_CALL",
        {"has_result": result_dict is not None, "duration": getattr(event, "duration", None), "exception": str(exception) if exception else None},
        invocation=ctx,
        tool_name=tool_name,
    )


def _register_hooks(agent: Any, forced_tool_name: Optional[str] = None) -> None:
    """Register current Strands hooks and optionally harden tool arguments."""

    agent.add_hook(on_before_invocation, BeforeInvocationEvent)
    agent.add_hook(on_after_invocation, AfterInvocationEvent)

    if forced_tool_name:
        def _force_arguments(event: Any) -> None:
            ctx = _context_invocation.get()
            if ctx is None:
                return

            tool_name = _event_tool_name(event)
            if tool_name != forced_tool_name:
                return

            fixed = _fixed_tool_arguments(forced_tool_name, ctx)
            if not fixed:
                return

            tool_use = getattr(event, "tool_use", None)
            if isinstance(tool_use, dict):
                current_input = tool_use.get("input")
                if isinstance(current_input, dict):
                    current_input.clear()
                    current_input.update(fixed)

        agent.add_hook(
            _force_arguments,
            BeforeToolCallEvent,
            order=-100,
        )

    agent.add_hook(on_before_tool_call, BeforeToolCallEvent)
    agent.add_hook(on_after_tool_call, AfterToolCallEvent)


def _fixed_tool_arguments(tool_name: str, ctx: AgentInvocationContext) -> Dict[str, Any]:
    """Return trusted tool inputs; model-generated infrastructure arguments are ignored."""
    if tool_name == "get_candidate_profile":
        return {"user_id": ctx.user_id}

    if tool_name == "normalize_opportunity":
        return {"opportunity": dict(ctx.opportunity_payload)}

    if tool_name == "analyze_candidate_match":
        return {
            "user_profile": ctx.tool_results.get("get_candidate_profile", {}),
            "opportunity": ctx.tool_results.get(
                "normalize_opportunity",
                ctx.opportunity_payload,
            ),
        }

    if tool_name == "re_evaluate_application":
        return {
            "user_id": ctx.user_id,
            "opportunity": ctx.tool_results.get(
                "normalize_opportunity",
                ctx.opportunity_payload,
            ),
            "clarification_key": ctx.clarification_key or "",
            "clarification_answer": None,
            "current_attempts": int(ctx.clarification_attempt),
        }

    if tool_name == "make_application_decision":
        analysis = (
            ctx.tool_results.get("analyze_candidate_match")
            if ctx.run_id == "RUN-1"
            else (ctx.tool_results.get("re_evaluate_application") or {}).get("match_analysis")
        )
        return {"match_analysis": analysis or {}}

    if tool_name == "generate_application_draft":
        return {
            "user_id": ctx.user_id,
            "opportunity": ctx.tool_results.get(
                "normalize_opportunity",
                ctx.opportunity_payload,
            ),
            "match_analysis": (
                ctx.tool_results.get("analyze_candidate_match")
                if ctx.run_id == "RUN-1"
                else (ctx.tool_results.get("re_evaluate_application") or {}).get("match_analysis")
            ),
        }

    if tool_name == "request_clarification":
        reason = (
            (ctx.decision_result or {}).get("reason")
            or "Clarification required"
        )
        field = (
            ctx.clarification_key
            or (ctx.decision_result or {}).get("clarification_key")
            or "cgpa"
        )
        return {
            "user_id": ctx.user_id,
            "field": field,
            "reason": reason,
        }

    return {}


def _build_agent(run_id: str, tools: List[Any], system_prompt: Optional[str] = None, forced_tool_name: Optional[str] = None) -> Any:
    """Build one fresh REAL Strands Agent.

    ApplyX deliberately creates one short-lived agent per workflow step and gives it
    exactly one allowed tool. This makes the workflow deterministic while the model
    still participates through the real Strands tool-calling loop.
    """
    if not STRANDS_AVAILABLE or Agent is None:
        raise RuntimeError("Strands Agents SDK is not installed. Install requirements.txt for REAL mode.")
    if not tools:
        raise ValueError("At least one tool is required.")

    provider = "gemini"
    model_id = "gemini-3.5-flash"
    model = None

    if provider != "gemini":
        raise RuntimeError(
            "ApplyX REAL mode is configured for Google Gemini only. "
            "Set STRANDS_PROVIDER=gemini."
        )

    try:
        from strands.models.gemini import GeminiModel
    except ImportError as exc:
        raise RuntimeError(
            'Gemini support is not installed. Run: python -m pip install -U "strands-agents[gemini]"'
        ) from exc

    # You may hardcode a NEW local key here, or provide GEMINI_API_KEY in the environment.
    api_key = (
        HARDCODED_GEMINI_API_KEY.strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Put it in HARDCODED_GEMINI_API_KEY "
            "or set GEMINI_API_KEY."
        )

    model = GeminiModel(
        client_args={"api_key": api_key},
        model_id=model_id or "gemini-3.6-flash",
        params={"temperature": 0.1, "max_output_tokens": 2048},
    )

    kwargs: Dict[str, Any] = {
        "name": f"ApplyX-{run_id}",
        "system_prompt": system_prompt or APPLYX_SYSTEM_PROMPT,
        "tools": tools,
    }
    if model is not None:
        kwargs["model"] = model

    agent = Agent(**kwargs)
    _register_hooks(agent, forced_tool_name=forced_tool_name)
    return agent


def _forced_tool_choice(tool_name: str) -> Dict[str, Any]:
    """Current Strands ToolChoice syntax for forcing one named tool."""
    return {"tool": {"name": tool_name}}


async def _run_real_agent_step(
    *,
    ctx: AgentInvocationContext,
    tool_fn: Any,
    tool_name: str,
    prompt: str,
) -> None:
    """Run exactly one REAL Strands tool-calling step.

    The model is given exactly one tool and Strands is instructed to force that tool.
    We fail closed if the SDK/model returns without actually executing it.
    """
    if tool_name in ctx.executed_tools:
        raise RuntimeError(f"Workflow violation: '{tool_name}' was already executed.")

    agent = _build_agent(
        ctx.run_id,
        [tool_fn],
        forced_tool_name=tool_name,
        system_prompt=(
            APPLYX_SYSTEM_PROMPT
            + "\n\nTHIS STEP IS STRICTLY CONTROLLED. You have exactly one available tool. "
            + f"You MUST call '{tool_name}' now, then stop. Do not answer with prose before calling it."
        ),
    )

    # One model turn is enough: the model must issue the forced tool call.
    # A second turn would allow the same single available tool to be requested again.
    limits = {"turns": 1}
    stream = agent.stream_async(
        prompt,
        limits=limits,
        tool_choice=_forced_tool_choice(tool_name),
        invocation_state={
            "run_id": ctx.run_id,
            "application_id": ctx.application_id,
            "user_id": ctx.user_id,
            "context": ctx,
        },
    )

    # Consume the complete stream. Hooks record the actual durable tool execution.
    # A complete tool call is therefore observable even when the model's final prose
    # is ignored by ApplyX.
    async for event in stream:
        terminal = extract_terminal_result(event)
        if terminal is not None:
            ctx.terminal_agent_result = terminal

    if ctx.cancellation_reason:
        return
    if tool_name not in ctx.executed_tools:
        raise RuntimeError(
            f"Strands step completed without executing required tool '{tool_name}'."
        )
    if ctx.tool_error:
        raise RuntimeError(ctx.tool_error)


def _run_mock_agent(user_id: str, application_id: str, raw_opportunity: Dict[str, Any], run_id: str) -> None:
    """Explicit mock backend used by benchmarks/tests; never called by REAL mode."""
    ctx = _context_invocation.get()
    if not ctx:
        raise RuntimeError("Mock execution requires invocation context.")
    profile = get_candidate_profile(user_id)
    on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "get_candidate_profile"})(), "cancel_tool": False})())
    on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "get_candidate_profile"})(), "result": profile, "duration": 0.001, "exception": None})())

    normalized = normalize_opportunity(raw_opportunity)
    on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "normalize_opportunity"})(), "cancel_tool": False})())
    on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "normalize_opportunity"})(), "result": normalized, "duration": 0.001, "exception": None})())

    if run_id == "RUN-1":
        analysis = analyze_candidate_match(profile, normalized)
        tool_name = "analyze_candidate_match"
    else:
        analysis_wrapper = re_evaluate_application(user_id, normalized, ctx.clarification_key or "", None, ctx.clarification_attempt)
        analysis = analysis_wrapper.get("match_analysis", {})
        tool_name = "re_evaluate_application"
    on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": tool_name})(), "cancel_tool": False})())
    on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": tool_name})(), "result": (analysis_wrapper if run_id == "RUN-2" else analysis), "duration": 0.001, "exception": None})())

    # For Run 2, make_application_decision reads the structured match_analysis nested in the tool result.
    decision = make_application_decision(analysis)
    on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "make_application_decision"})(), "cancel_tool": False})())
    on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "make_application_decision"})(), "result": decision, "duration": 0.001, "exception": None})())

    action = decision.get("action")
    if action == "APPLY":
        draft = generate_application_draft(user_id, normalized, analysis)
        on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "generate_application_draft"})(), "cancel_tool": False})())
        on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "generate_application_draft"})(), "result": draft, "duration": 0.001, "exception": None})())
    elif action == "REVIEW":
        field = ctx.clarification_key or "cgpa"
        req = request_clarification(user_id, field, decision.get("reason", "Clarification required"))
        on_before_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "request_clarification"})(), "cancel_tool": False})())
        on_after_tool_call(type("E", (), {"selected_tool": type("T", (), {"name": "request_clarification"})(), "result": req, "duration": 0.001, "exception": None})())


async def run_strands_application_flow(
    user_id: str,
    raw_opportunity: Any,
    application_id: str,
    orchestrator=None,
    run_id: str = "RUN-1",
    db_path: Optional[str] = None,
    clarification_id: Optional[str] = None,
    clarification_key: Optional[str] = None,
    clarification_attempt: int = 0,
    agent_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if orchestrator is None:
        raise RuntimeError("No orchestrator bound to current agent invocation")
    if run_id not in {"RUN-1", "RUN-2"}:
        raise ValueError("run_id must be RUN-1 or RUN-2")

    mode = (agent_mode or os.getenv("APPLYX_AGENT_MODE", "REAL")).strip().upper()
    if mode not in {"REAL", "MOCK"}:
        raise ValueError("APPLYX_AGENT_MODE must be REAL or MOCK.")

    opp_dict = raw_opportunity.to_dict() if hasattr(raw_opportunity, "to_dict") else dict(raw_opportunity or {})
    ctx = AgentInvocationContext(
        run_id=run_id,
        application_id=application_id,
        user_id=user_id,
        db_path=db_path or orchestrator.db_path,
        opportunity_payload=opp_dict,
        clarification_id=clarification_id,
        clarification_key=clarification_key,
        clarification_attempt=clarification_attempt,
    )

    clear_invocation_traces(application_id, run_id)
    orch_token = set_orchestrator(orchestrator)
    ctx_token = _context_invocation.set(ctx)
    try:
        record_agent_trace_event("STRANDS_AGENT_STARTED", {"mode": mode}, invocation=ctx)

        if mode == "MOCK":
            _run_mock_agent(user_id, application_id, opp_dict, run_id)
        else:
            if not STRANDS_AVAILABLE:
                raise RuntimeError("REAL mode requested but the Strands Agents SDK is unavailable.")

            normalized_prompt = json.dumps(ctx.opportunity_payload, ensure_ascii=False, default=str)

            await _run_real_agent_step(
                ctx=ctx,
                tool_fn=get_candidate_profile,
                tool_name="get_candidate_profile",
                prompt=(
                    f"Call get_candidate_profile for candidate {ctx.user_id}. "
                    "The invocation context is trusted; return the candidate profile."
                ),
            )

            await _run_real_agent_step(
                ctx=ctx,
                tool_fn=normalize_opportunity,
                tool_name="normalize_opportunity",
                prompt=(
                    "Call normalize_opportunity for this opportunity. "
                    f"Opportunity payload: {normalized_prompt}"
                ),
            )

            profile = ctx.tool_results.get("get_candidate_profile", {})
            normalized = ctx.tool_results.get("normalize_opportunity", ctx.opportunity_payload)

            if run_id == "RUN-1":
                await _run_real_agent_step(
                    ctx=ctx,
                    tool_fn=analyze_candidate_match,
                    tool_name="analyze_candidate_match",
                    prompt=(
                        "Call analyze_candidate_match using the candidate profile and normalized opportunity below.\n"
                        f"Candidate profile: {json.dumps(profile, ensure_ascii=False, default=str)}\n"
                        f"Normalized opportunity: {json.dumps(normalized, ensure_ascii=False, default=str)}"
                    ),
                )
            else:
                await _run_real_agent_step(
                    ctx=ctx,
                    tool_fn=re_evaluate_application,
                    tool_name="re_evaluate_application",
                    prompt=(
                        "Call re_evaluate_application. Do not mutate the Knowledge Base. "
                        f"Candidate: {ctx.user_id}. Clarification field: {ctx.clarification_key or ''}. "
                        f"Clarification attempt: {ctx.clarification_attempt}.\n"
                        f"Normalized opportunity: {json.dumps(normalized, ensure_ascii=False, default=str)}"
                    ),
                )

            analysis = (
                ctx.tool_results.get("analyze_candidate_match")
                if run_id == "RUN-1"
                else (ctx.tool_results.get("re_evaluate_application") or {}).get("match_analysis")
            )
            if not isinstance(analysis, dict):
                raise RuntimeError("Required deterministic match analysis was not produced.")

            await _run_real_agent_step(
                ctx=ctx,
                tool_fn=make_application_decision,
                tool_name="make_application_decision",
                prompt=(
                    "Call make_application_decision using this deterministic match analysis. "
                    "Do not invent, edit, or reinterpret it.\n"
                    f"Match analysis: {json.dumps(analysis, ensure_ascii=False, default=str)}"
                ),
            )

            if not isinstance(ctx.decision_result, dict):
                raise RuntimeError("make_application_decision did not return a structured result.")

            action = ctx.decision_result.get("action")
            if action == "APPLY":
                await _run_real_agent_step(
                    ctx=ctx,
                    tool_fn=generate_application_draft,
                    tool_name="generate_application_draft",
                    prompt=(
                        "Call generate_application_draft exactly once. "
                        f"Candidate: {ctx.user_id}.\n"
                        f"Normalized opportunity: {json.dumps(normalized, ensure_ascii=False, default=str)}\n"
                        f"Deterministic match analysis: {json.dumps(analysis, ensure_ascii=False, default=str)}"
                    ),
                )
            elif action == "REVIEW":
                field = ctx.clarification_key or ctx.decision_result.get("clarification_key") or "cgpa"
                reason = ctx.decision_result.get("reason") or "Clarification required"
                await _run_real_agent_step(
                    ctx=ctx,
                    tool_fn=request_clarification,
                    tool_name="request_clarification",
                    prompt=(
                        "Call request_clarification exactly once. "
                        f"Candidate: {ctx.user_id}. Field: {field}. Reason: {reason}"
                    ),
                )
            elif action == "SKIP":
                pass
            else:
                raise RuntimeError(f"Invalid structured action from decision tool: {action!r}")


        traces = get_invocation_traces(application_id, run_id)

        if ctx.cancellation_reason == "TOOL_LIMIT":
            return {"status": "FAILED", "terminal_outcome": "TOOL_LIMIT", "reason": "AGENT_TOOL_BUDGET_EXCEEDED", "application_id": application_id, "traces": traces}
        if ctx.cancellation_reason == "WORKFLOW_VIOLATION":
            return {"status": "FAILED", "terminal_outcome": "INCOMPLETE_WORKFLOW", "reason": "Duplicate tool invocation was blocked by the workflow guardrail.", "application_id": application_id, "traces": traces}
        if ctx.tool_error:
            return {"status": "FAILED", "terminal_outcome": "TOOL_ERROR", "reason": f"Agent tool error: {ctx.tool_error}", "application_id": application_id, "traces": traces}

        stop_reason = None
        terminal = ctx.terminal_agent_result
        if terminal is not None:
            stop_reason = getattr(terminal, "stop_reason", None)
            if stop_reason is None and isinstance(terminal, dict):
                stop_reason = terminal.get("stop_reason")
            stop_reason = str(stop_reason or "")

        if stop_reason in {"cancelled", "canceled"}:
            return {"status": "FAILED", "terminal_outcome": "CANCELLED", "reason": "AGENT_CANCELLED", "application_id": application_id, "traces": traces}

        expected_prefix = [
            "get_candidate_profile",
            "normalize_opportunity",
            "re_evaluate_application" if run_id == "RUN-2" else "analyze_candidate_match",
            "make_application_decision",
        ]
        action = ctx.decision_result.get("action") if ctx.decision_result else None
        expected_sequence = expected_prefix + (["generate_application_draft"] if action == "APPLY" else ["request_clarification"] if action == "REVIEW" else [])

        if ctx.executed_tools != expected_sequence:
            return {
                "status": "FAILED",
                "terminal_outcome": "INCOMPLETE_WORKFLOW",
                "reason": f"Tool sequence violation. Expected {expected_sequence}, got {ctx.executed_tools}.",
                "application_id": application_id,
                "traces": traces,
            }

        if not ctx.decision_result or action not in {"APPLY", "REVIEW", "SKIP"}:
            return {"status": "FAILED", "terminal_outcome": "INCOMPLETE_WORKFLOW", "reason": "make_application_decision did not return a valid structured action.", "application_id": application_id, "traces": traces}

        if action == "APPLY":
            draft_result = ctx.tool_results.get("generate_application_draft")
            if not isinstance(draft_result, dict) or draft_result.get("status") not in {"DRAFT_CREATED", "ALREADY_EXISTS"}:
                return {"status": "FAILED", "terminal_outcome": "INCOMPLETE_WORKFLOW", "reason": "APPLY branch did not successfully persist Draft V1.", "application_id": application_id, "traces": traces}
        elif action == "REVIEW":
            clarification_result = ctx.tool_results.get("request_clarification")
            if not isinstance(clarification_result, dict) or not clarification_result.get("clarification_id"):
                return {"status": "FAILED", "terminal_outcome": "INCOMPLETE_WORKFLOW", "reason": "REVIEW branch did not create a clarification request.", "application_id": application_id, "traces": traces}

        analysis = ctx.tool_results.get("analyze_candidate_match")
        if run_id == "RUN-2":
            wrapper = ctx.tool_results.get("re_evaluate_application") or {}
            analysis = wrapper.get("match_analysis") if isinstance(wrapper, dict) else None
        analysis = analysis or {}
        decision = ctx.decision_result

        return {
            "status": "COMPLETED",
            "terminal_outcome": "COMPLETED",
            "decision": action,
            "application_id": application_id,
            "opportunity_id": opp_dict.get("id"),
            "confidence": decision.get("confidence", 0.85),
            "fit_score": decision.get("fit_score", 0),
            "evidence_coverage": analysis.get("evidence_coverage", 0.0) if isinstance(analysis, dict) else 0.0,
            "risk": decision.get("risk", "UNKNOWN"),
            "reason": decision.get("reason", "Evaluation complete"),
            "clarification_prompt": decision.get("clarification_prompt"),
            "clarification_id": ctx.clarification_id,
            "match_analysis": analysis,
            "traces": traces,
        }
    except Exception as exc:
        record_agent_trace_event("AGENT_EXECUTION_EXCEPTION", {"error": str(exc)}, invocation=ctx)
        raise
    finally:
        _context_invocation.reset(ctx_token)
        reset_orchestrator(orch_token)
