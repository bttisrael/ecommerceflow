from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
    from crewai import LLM
except Exception as exc:
    raise RuntimeError(
        "CrewAI is required for this cloud runner. Install crewai and anthropic."
    ) from exc


@tool("run_commerceflow_training_pipeline")
def run_commerceflow_training_pipeline(_: str = "") -> str:
    """
    Runs the deterministic CommerceFlow delay-risk ML pipeline using BigQuery as source,
    then uploads artifacts to GCS when a bucket is configured.
    """
    bq_table = os.getenv(
        "BQ_FEATURE_TABLE",
        "otimizador-cargas.commerce_gold.delay_risk_features",
    )
    max_rows = os.getenv("MAX_ROWS", "100000")
    gcs_bucket = os.getenv("GCS_BUCKET", "")
    gcs_prefix = os.getenv("GCS_PREFIX", "commerceflow/ml_runs/latest")

    cmd = [
        sys.executable,
        "commerceflow_delay_risk_crewai_v6.py",
        "--source",
        "bigquery",
        "--bq-table",
        bq_table,
        "--max-rows",
        max_rows,
        "--direct",
        "--no-git",
        "--no-xgboost",
        "--no-lightgbm",
    ]

    if gcs_bucket:
        cmd += ["--gcs-bucket", gcs_bucket, "--gcs-prefix", gcs_prefix]

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(Path(__file__).resolve().parent),
    )

    output = (
        "STDOUT:\n"
        + result.stdout[-12000:]
        + "\n\nSTDERR:\n"
        + result.stderr[-12000:]
    )

    if result.returncode != 0:
        return f"PIPELINE_ERROR\nReturn code: {result.returncode}\n{output}"

    return f"PIPELINE_SUCCESS\nReturn code: {result.returncode}\n{output}"


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    llm = LLM(
        model=os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-5"),
        api_key=api_key,
        temperature=0.0,
    )

    orchestrator = Agent(
        role="CommerceFlow Cloud ML Orchestrator",
        goal=(
            "Run the CommerceFlow BigQuery-based delay-risk training pipeline, "
            "verify that it completes successfully, and ensure model artifacts are uploaded."
        ),
        backstory=(
            "You are a senior ML platform orchestrator. You coordinate deterministic "
            "data science tools in cloud environments. You do not rewrite the model logic; "
            "you execute and validate the production training pipeline."
        ),
        tools=[run_commerceflow_training_pipeline],
        llm=llm,
        verbose=True,
        max_iter=3,
    )

    task = Task(
        description=(
            "Call run_commerceflow_training_pipeline. "
            "The task is complete only if the tool returns PIPELINE_SUCCESS. "
            "Summarize the key result: BigQuery source, model artifact, metrics, dashboard, and GCS upload."
        ),
        agent=orchestrator,
        expected_output=(
            "A concise execution report confirming whether the cloud ML training pipeline succeeded."
        ),
    )

    crew = Crew(
        agents=[orchestrator],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()