"""
CommerceFlow AI — CrewAI Orchestration Baseline
==============================================

This file follows the same architectural idea as the user's multi_agent_ds_v8:
- CrewAI orchestrates agents sequentially
- each agent has one focused tool
- Python performs deterministic work
- optional Claude/CrewAI layer provides reasoning and execution flow

MVP agents:
1. Website Builder       -> validates frontend/backend files exist
2. API/BigQuery Agent    -> validates API and BigQuery config
3. Data Quality Agent    -> checks local order database
4. ML Scientist Agent    -> trains delay risk model
5. Dashboard Agent       -> validates dashboard artifacts
6. GitOps Agent          -> commits and optionally pushes project
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

llm_agent = LLM(
    model=os.getenv("CREWAI_MODEL", "anthropic/claude-sonnet-4-5"),
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.0,
)


def run_cmd(cmd: list[str], cwd: Path = ROOT) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, errors="replace")
    return (p.stdout or "") + (p.stderr or "")


@tool("validate_website_project")
def validate_website_project(_: str = "") -> str:
    required = [
        ROOT / "frontend" / "src" / "App.jsx",
        ROOT / "frontend" / "src" / "style.css",
        ROOT / "backend" / "main.py",
        ROOT / "backend" / "order_generator.py",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        return "WEBSITE_VALIDATION_ERROR: missing files: " + ", ".join(missing)
    return "WEBSITE_VALIDATION_SUCCESS: React frontend and FastAPI backend files are present."


@tool("validate_api_bigquery_config")
def validate_api_bigquery_config(_: str = "") -> str:
    table = os.getenv("BIGQUERY_ORDERS_TABLE")
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    status = {
        "bigquery_table_configured": bool(table),
        "bigquery_table": table,
        "google_credentials_configured": bool(creds),
        "credentials_path_exists": bool(creds and Path(creds).exists()),
        "raw_schema_file": str((ROOT / "sql" / "create_raw_tables.sql").relative_to(ROOT)),
    }
    return "BIGQUERY_CONFIG_CHECK\n" + json.dumps(status, indent=2)


@tool("run_data_quality_checks")
def run_data_quality_checks(_: str = "") -> str:
    import pandas as pd
    from backend.database import export_orders_dataframe

    df = export_orders_dataframe()
    if df.empty:
        return "DATA_QUALITY_WARNING: no orders found. Start backend and generate orders first."
    report = {
        "rows": int(len(df)),
        "duplicate_order_id": int(df["order_id"].duplicated().sum()),
        "missing_email_pct": float(df["customer_email"].isna().mean()),
        "negative_order_values": int((df["order_value"] <= 0).sum()),
        "invalid_distance": int((df["distance_km"] < 0).sum()),
        "delay_rate": float(df["delay_risk_label"].mean()),
        "states": int(df["state"].nunique()),
        "carriers": int(df["carrier"].nunique()),
    }
    out = ROOT / "docs" / "data_quality_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return "DATA_QUALITY_SUCCESS\n" + json.dumps(report, indent=2)


@tool("train_delay_risk_model_tool")
def train_delay_risk_model_tool(_: str = "") -> str:
    result = run_cmd(["python", "ml/train_delay_model.py"])
    return "ML_TRAINING_RESULT\n" + result[-3000:]


@tool("validate_dashboard_artifacts")
def validate_dashboard_artifacts(_: str = "") -> str:
    required = [ROOT / "dashboard" / "streamlit_app.py"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        return "DASHBOARD_ERROR: missing " + ", ".join(missing)
    return "DASHBOARD_SUCCESS: Streamlit dashboard exists. Run: streamlit run dashboard/streamlit_app.py"


@tool("gitops_commit_push")
def gitops_commit_push(_: str = "") -> str:
    if not (ROOT / ".git").exists():
        run_cmd(["git", "init"])
    run_cmd(["git", "add", "."])
    status = run_cmd(["git", "status", "--porcelain"])
    if status.strip():
        commit = run_cmd(["git", "commit", "-m", "Refresh CommerceFlow AI project"])
    else:
        commit = "No changes to commit."
    if os.getenv("GIT_PUSH", "false").lower() == "true":
        pull = run_cmd(["git", "pull", "--rebase", "origin", "main"])
        push = run_cmd(["git", "push", "origin", "main"])
        return f"GITOPS_SUCCESS\n{commit}\n{pull}\n{push}"
    return f"GITOPS_SUCCESS_LOCAL_ONLY\n{commit}\nSet GIT_PUSH=true to push automatically."


website_agent = Agent(
    role="Website Builder Agent",
    goal="Validate and maintain the e-commerce simulator website and FastAPI backend structure.",
    backstory="You are a senior full-stack engineer specialized in data product prototypes.",
    tools=[validate_website_project],
    llm=llm_agent,
    max_iter=2,
)

api_agent = Agent(
    role="API and BigQuery Ingestion Agent",
    goal="Validate API-to-BigQuery readiness for raw order ingestion.",
    backstory="You are a data engineer focused on event ingestion and BigQuery raw layers.",
    tools=[validate_api_bigquery_config],
    llm=llm_agent,
    max_iter=2,
)

data_quality_agent = Agent(
    role="Data Quality Agent",
    goal="Check generated orders for quality issues before model training.",
    backstory="You are a data quality engineer validating operational datasets.",
    tools=[run_data_quality_checks],
    llm=llm_agent,
    max_iter=2,
)

ml_agent = Agent(
    role="ML Scientist Agent",
    goal="Train a delay risk model using generated order data.",
    backstory="You are a machine learning scientist focused on logistics delay risk prediction.",
    tools=[train_delay_risk_model_tool],
    llm=llm_agent,
    max_iter=2,
)

dashboard_agent = Agent(
    role="Dashboard Agent",
    goal="Validate monitoring dashboards for order stats and delay risk.",
    backstory="You are a data product engineer building decision-support dashboards.",
    tools=[validate_dashboard_artifacts],
    llm=llm_agent,
    max_iter=2,
)

git_agent = Agent(
    role="GitOps Agent",
    goal="Commit and optionally push the project after each run.",
    backstory="You automate clean portfolio publishing workflows.",
    tools=[gitops_commit_push],
    llm=llm_agent,
    max_iter=2,
)


def build_crew() -> Crew:
    tasks = [
        Task(description="Validate the website/backend project structure.", expected_output="Website validation result.", agent=website_agent),
        Task(description="Validate BigQuery ingestion configuration and raw schema readiness.", expected_output="BigQuery configuration report.", agent=api_agent),
        Task(description="Run data quality checks on generated local orders.", expected_output="Data quality report.", agent=data_quality_agent),
        Task(description="Train the delay risk model if enough generated data exists.", expected_output="Model training result and metrics.", agent=ml_agent),
        Task(description="Validate dashboard artifacts.", expected_output="Dashboard validation result.", agent=dashboard_agent),
        Task(description="Commit and optionally push project changes.", expected_output="GitOps result.", agent=git_agent),
    ]
    return Crew(agents=[website_agent, api_agent, data_quality_agent, ml_agent, dashboard_agent, git_agent], tasks=tasks, process=Process.sequential, verbose=True)


if __name__ == "__main__":
    print(build_crew().kickoff())
