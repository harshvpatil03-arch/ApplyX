# ApplyX — REAL Hackathon Setup

ApplyX uses **Amazon Bedrock + Strands Agents** as the primary REAL provider for the Agents for Humans hackathon.

## Activate
```powershell
.\.venv\Scripts\Activate.ps1
```

## Install
```powershell
python -m pip install -r requirements.txt
```

## Configure AWS
```powershell
aws configure
```

or:
```powershell
aws login
```

Never place AWS secrets in source code or chat.

## Runtime configuration
```powershell
$env:STRANDS_PROVIDER="bedrock"
$env:AWS_REGION="us-east-1"
$env:STRANDS_MODEL_ID="global.anthropic.claude-sonnet-4-6"
$env:APPLYX_AGENT_MODE="REAL"
```

Make sure the selected Bedrock model is enabled in the account/region.

## Preflight
```powershell
python diagnostics/check_real_provider.py
```

## Start
```powershell
python -m streamlit run app.py
```

## Verification
```powershell
python -m unittest tests/test_core.py
python diagnostics/run_benchmark.py --mode=deterministic
python diagnostics/run_benchmark.py --mode=agent
```

Expected local deterministic results: **23/23 tests** and **30/30 Mode A cases**. With valid AWS/Bedrock access, the REAL agent harness should produce **5/5**.
