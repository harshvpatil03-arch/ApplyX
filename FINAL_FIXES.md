# ApplyX FINAL REAL FIX

## What was fixed
1. REAL Strands workflow no longer relies on the LLM obeying a prose-only ordering instruction.
   Each workflow step creates a fresh real Strands Agent with exactly one allowed tool and a forced ToolChoice for that tool.
2. Streamlit/SQLite contention was hardened with WAL, busy timeout, thread-safe connections, and one-time-per-process schema initialization.
3. The existing architecture, UI layout/theme, FSM, draft/approval/submission boundaries, and MOCK backend were preserved.
4. `requirements.txt` pins `strands-agents==1.55.0`, matching the current PyPI release used for the implementation.

## Verification performed in the build environment
- `python -m py_compile app.py core/*.py diagnostics/run_benchmark.py tests/test_core.py`
- `python -m unittest tests/test_core.py` -> 23/23 passed
- `APPLYX_AGENT_MODE=MOCK python diagnostics/run_benchmark.py --mode=mock` -> 5/5 passed

## Important REAL-mode requirement
The build environment used for verification did not have cloud credentials or the Strands SDK installed, so a live provider request was not executed here. REAL mode therefore still requires your configured provider credentials and `pip install -r requirements.txt` in your `.venv`.
