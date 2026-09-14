# ApplyX — REAL Hackathon Build

Primary REAL provider: **Amazon Bedrock + Strands Agents**.

### Start
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
aws configure
$env:STRANDS_PROVIDER="bedrock"
$env:AWS_REGION="us-east-1"
$env:STRANDS_MODEL_ID="global.anthropic.claude-sonnet-4-6"
$env:APPLYX_AGENT_MODE="REAL"
python diagnostics/check_real_provider.py
python -m streamlit run app.py
```

### Verification
```powershell
python -m unittest tests/test_core.py
python diagnostics/run_benchmark.py --mode=deterministic
python diagnostics/run_benchmark.py --mode=agent
```

MOCK remains available only as an explicit deterministic development backend; it is not the hackathon demo mode.
