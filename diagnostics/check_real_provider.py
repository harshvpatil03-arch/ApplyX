"""ApplyX REAL provider preflight. Checks local configuration only."""
from __future__ import annotations
import os

def main() -> int:
    provider = os.getenv("STRANDS_PROVIDER", "bedrock").strip().lower()
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    model_id = os.getenv("STRANDS_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
    print(f"Provider: {provider}")
    print(f"Region:   {region}")
    print(f"Model:    {model_id}")
    if provider != "bedrock":
        print("FAIL: This hackathon build uses Amazon Bedrock.")
        return 1
    try:
        import boto3
    except Exception as exc:
        print(f"FAIL: boto3 unavailable: {exc}")
        return 1
    try:
        session = boto3.Session(region_name=region)
        creds = session.get_credentials()
        if creds is None and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            print("FAIL: AWS credentials were not found.")
            print("Run `aws configure` or `aws login`, then rerun this check.")
            return 2
        print("PASS: AWS credential source detected (secrets not shown).")
        print("NEXT: ensure Bedrock model access is enabled for the selected model/region.")
        return 0
    except Exception as exc:
        print(f"FAIL: Could not resolve AWS credentials: {exc}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
