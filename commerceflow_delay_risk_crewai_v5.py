"""
CommerceFlow AI — CrewAI Delay-Risk Data Science Pipeline
=========================================================

Adapted from multi_agent_ds_v8 style: CrewAI sequential orchestration, one tool per
agent, deterministic Python for the heavy work, and no Telegram-bot deployer.

Pipeline:
1. Ingestion              -> SQLite orders table to silver parquet
2. Data Analysis          -> quality, leakage, target distribution
3. Feature Engineering    -> logistics/date/risk features
4. EDA                    -> charts + leakage-free ML-ready table
5. Hypothesis Validation  -> business hypotheses and delay-rate lifts
6. ML                     -> model competition + model.pkl
7. Business Performance   -> cost/value simulation
8. Notebook               -> technical notebook
9. HTML Dashboard         -> static stakeholder dashboard
10. GitOps                -> commit and push code/reports to GitHub

Run:
    python commerceflow_delay_risk_crewai.py --direct
    python commerceflow_delay_risk_crewai.py --max-rows 500000 --direct

CrewAI mode:
    python commerceflow_delay_risk_crewai.py

If CrewAI or ANTHROPIC_API_KEY is not configured, use --direct.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sqlite3
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sklearn.inspection import permutation_importance
except Exception:
    permutation_importance = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False
    def tool(name: str):
        def wrapper(fn):
            return fn
        return wrapper

load_dotenv()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("CommerceFlowDS")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler("commerceflow_ds_pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

logger = setup_logger()


@dataclass
class Config:
    project_name: str = "CommerceFlow AI — Delay Risk ML Pipeline"
    db_path: str = "commerceflow.db"
    table_name: str = "orders"
    target_col: str = "delay_risk_label"
    max_rows: int = 500_000
    random_state: int = 42
    output_dir: str = "ml_outputs"
    test_size: float = 0.20
    n_jobs: int = 2
    max_cat_cardinality: int = 80
    sample_for_importance: int = 12_000
    cost_false_negative: float = 120.0
    cost_false_positive: float = 15.0
    value_true_positive: float = 80.0
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    anthropic_model: str = "anthropic/claude-sonnet-4-5"

    # GitOps
    git_enabled: bool = True
    git_remote_name: str = "origin"
    git_branch: str = "main"
    git_push: bool = True
    git_track_large_artifacts: bool = False
    git_commit_prefix: str = "Auto-refresh CommerceFlow delay-risk pipeline"

    def paths(self) -> Dict[str, str]:
        out = self.output_dir
        return {
            "silver": f"{out}/df1_silver_orders.parquet",
            "analysis_md": f"{out}/Data_Quality_Report.md",
            "features": f"{out}/df2_features.parquet",
            "ml_ready": f"{out}/df3_ml_ready.parquet",
            "predictions": f"{out}/df4_predictions.parquet",
            "model": f"{out}/model.pkl",
            "metrics_json": f"{out}/model_metrics.json",
            "metrics_md": f"{out}/Model_Metrics.md",
            "feature_importance_png": f"{out}/feature_importance.png",
            "error_md": f"{out}/Error_Analysis.md",
            "hyp_json": f"{out}/hypothesis_results.json",
            "hyp_md": f"{out}/Hypothesis_Validation.md",
            "hyp_png": f"{out}/hypothesis_lift.png",
            "business_md": f"{out}/Business_Performance.md",
            "business_png": f"{out}/business_performance.png",
            "notebook": f"{out}/analysis_notebook.ipynb",
            "html": f"{out}/dashboard.html",
            "run_results": f"{out}/pipeline_run_results.json",
        }

CONFIG = Config()
P = CONFIG.paths()


def refresh_paths() -> None:
    global P
    P = CONFIG.paths()


def ensure_out() -> None:
    Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)


def json_safe(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(str(type(obj)))


def save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_safe), encoding="utf-8")


def load_json(path: str, default: Any = None) -> Any:
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data available._"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception:
        return df.head(max_rows).to_string(index=False)


def append_result(step: str, result: str) -> None:
    results = load_json(P["run_results"], [])
    results.append({"step": step, "result": result[:4000]})
    save_json(P["run_results"], results)


def collect_project_code_files(max_file_chars: int = 80_000) -> List[Dict[str, str]]:
    # Collects project source files for the generated notebook.
    root = Path(".").resolve()
    include_ext = {".py", ".sql", ".md", ".txt", ".json", ".jsx", ".js", ".css", ".html", ".example"}
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".ipynb_checkpoints"}
    skip_suffixes = {".parquet", ".pkl", ".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gz", ".7z", ".log", ".lock"}

    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & skip_dirs:
            continue
        if p.suffix.lower() in skip_suffixes:
            continue
        if p.suffix.lower() not in include_ext and p.name not in {".env.example", ".gitignore"}:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(content) > max_file_chars:
            content = content[:max_file_chars] + "\n\n# --- FILE TRUNCATED IN NOTEBOOK FOR READABILITY ---\n"
        files.append({"path": str(rel).replace("\\", "/"), "language": p.suffix.lower().replace(".", "") or "text", "content": content})
    return files


def write_project_code_manifest() -> str:
    files = collect_project_code_files()
    manifest_path = Path(CONFIG.output_dir) / "project_code_manifest.json"
    save_json(str(manifest_path), files)
    return str(manifest_path)


@tool("ingest_orders_from_sqlite")
def ingest_orders_from_sqlite(_: str = "") -> str:
    """Ingests CommerceFlow orders from SQLite and saves the silver parquet dataset."""
    try:
        ensure_out()
        db = Path(CONFIG.db_path)
        if not db.exists():
            return f"INGESTION_ERROR: database not found: {db.resolve()}"
        conn = sqlite3.connect(str(db))
        total = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {CONFIG.table_name}", conn)["n"].iloc[0])
        limit_sql = f" LIMIT {CONFIG.max_rows}" if CONFIG.max_rows and total > CONFIG.max_rows else ""
        query = f"SELECT * FROM {CONFIG.table_name} ORDER BY order_timestamp {limit_sql}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "", regex=True))
        for c in ["order_timestamp", "ingestion_timestamp"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if CONFIG.target_col in df.columns:
            df[CONFIG.target_col] = pd.to_numeric(df[CONFIG.target_col], errors="coerce").fillna(0).astype(int)
        df.to_parquet(P["silver"], index=False)
        out = f"INGESTION_SUCCESS\nSource rows: {total:,}\nIngested rows: {len(df):,}\nColumns: {len(df.columns)}\nFile: {P['silver']}"
        append_result("ingestion", out)
        return out
    except Exception as e:
        return f"INGESTION_ERROR: {e}\n{traceback.format_exc()}"


@tool("analyze_data_quality")
def analyze_data_quality(_: str = "") -> str:
    """Analyzes data quality, target distribution and leakage risks."""
    try:
        df = pd.read_parquet(P["silver"])
        target = CONFIG.target_col
        target_rate = float(df[target].mean()) if target in df.columns else None
        dtypes = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            "null_pct": [float(df[c].isna().mean() * 100) for c in df.columns],
        })
        leakage = [c for c in ["delivery_status", "actual_delivery_days", "delay_probability_true"] if c in df.columns]
        date_range = "N/A"
        if "order_timestamp" in df.columns:
            date_range = f"{df['order_timestamp'].min()} → {df['order_timestamp'].max()}"
        numeric_desc = df.select_dtypes(include="number").describe().T.reset_index().rename(columns={"index": "column"})
        lines = [
            "# CommerceFlow AI — Data Quality Report\n",
            f"Rows: **{len(df):,}**  \nColumns: **{len(df.columns):,}**  \nDate range: **{date_range}**  ",
            f"Target: `{target}`  \nDelay rate: **{target_rate:.2%}**" if target_rate is not None else "Target not found.",
            "\n## Leakage Columns to Exclude\n" + (", ".join(f"`{c}`" for c in leakage) if leakage else "None detected."),
            "\n## Column Audit\n" + md_table(dtypes, 100),
            "\n## Numeric Summary\n" + md_table(numeric_desc, 60),
            "\n## Business Reading\nThis dataset is suitable for a supervised delay-risk classification model. The model must use features known before delivery completion and must not use actual delivery outcome columns.",
        ]
        Path(P["analysis_md"]).write_text("\n\n".join(lines), encoding="utf-8")
        out = f"ANALYSIS_SUCCESS\nRows: {len(df):,}\nDelay rate: {target_rate:.2%}\nLeakage columns: {leakage}\nFile: {P['analysis_md']}"
        append_result("analysis", out)
        return out
    except Exception as e:
        return f"ANALYSIS_ERROR: {e}\n{traceback.format_exc()}"


@tool("build_features")
def build_features(_: str = "") -> str:
    """Builds deterministic logistics and time-based features for delay-risk prediction."""
    try:
        df = pd.read_parquet(P["silver"])
        target = CONFIG.target_col
        df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
        df["order_date"] = df["order_timestamp"].dt.date.astype(str)
        df["order_year"] = df["order_timestamp"].dt.year
        df["order_month"] = df["order_timestamp"].dt.month
        df["order_day"] = df["order_timestamp"].dt.day
        df["order_day_of_week"] = df["order_timestamp"].dt.dayofweek
        df["order_hour"] = df["order_timestamp"].dt.hour
        df["is_weekend"] = (df["order_day_of_week"] >= 5).astype(int)
        df["is_peak_hour"] = df["order_hour"].isin([8, 9, 10, 11, 12, 18, 19, 20, 21]).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["order_day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["order_day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12)
        df["same_state_dc"] = (df["state"].astype(str) == df["dc_state"].astype(str)).astype(int)
        df["estimated_minus_promised_days"] = df["estimated_delivery_days"] - df["promised_delivery_days"]
        df["estimated_over_promised"] = df["estimated_delivery_days"] / (df["promised_delivery_days"] + 1e-6)
        df["log_order_value"] = np.log1p(pd.to_numeric(df["order_value"], errors="coerce").fillna(0))
        df["log_distance_km"] = np.log1p(pd.to_numeric(df["distance_km"], errors="coerce").fillna(0))
        df["log_weight_kg"] = np.log1p(pd.to_numeric(df["order_weight_kg"], errors="coerce").fillna(0))
        df["value_per_kg"] = df["order_value"] / (df["order_weight_kg"] + 1e-6)
        df["value_per_km"] = df["order_value"] / (df["distance_km"] + 1e-6)
        df["weight_distance_interaction"] = df["order_weight_kg"] * df["distance_km"]
        df["distance_band"] = pd.cut(df["distance_km"], [-1, 80, 300, 900, 1800, np.inf], labels=["same_city", "short", "regional", "long", "very_long"]).astype(str)
        df["weight_band"] = pd.cut(df["order_weight_kg"], [-1, 0.5, 2, 5, 10, np.inf], labels=["very_light", "light", "medium", "heavy", "very_heavy"]).astype(str)
        traffic_rank = {"low": 0, "medium": 1, "high": 2, "severe": 3}
        weather_rank = {"clear": 0, "fog": 1, "heatwave": 1, "rain": 2, "storm": 3}
        route_rank = {"urban": 0, "regional": 1, "long_haul": 2, "remote": 3}
        df["traffic_risk_score"] = df["traffic_condition"].map(traffic_rank).fillna(1).astype(float)
        df["weather_risk_score"] = df["weather_condition"].map(weather_rank).fillna(0).astype(float)
        df["route_risk_score"] = df["route_type"].map(route_rank).fillna(1).astype(float)
        df["combined_operational_risk"] = df["traffic_risk_score"] + df["weather_risk_score"] + df["route_risk_score"] + df["carrier_base_delay_rate"].fillna(0) * 10 + (df["estimated_minus_promised_days"] > 0).astype(int)
        df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
        df.to_parquet(P["features"], index=False)
        out = f"FEATURES_SUCCESS\nRows: {len(df):,}\nColumns: {len(df.columns):,}\nFile: {P['features']}"
        append_result("features", out)
        return out
    except Exception as e:
        return f"FEATURES_ERROR: {e}\n{traceback.format_exc()}"


@tool("generate_eda_and_ml_ready")
def generate_eda_and_ml_ready(_: str = "") -> str:
    """Generates EDA charts and saves a leakage-free ML-ready table."""
    try:
        df = pd.read_parquet(P["features"])
        target = CONFIG.target_col
        # Charts
        plt.figure(figsize=(7, 4)); df[target].value_counts(normalize=True).sort_index().plot(kind="bar"); plt.title("Delay Risk Label Distribution"); plt.ylabel("Share"); plt.tight_layout(); plt.savefig(Path(CONFIG.output_dir)/"target_distribution.png", dpi=150); plt.close()
        for col, fname in [("traffic_condition","delay_by_traffic.png"),("weather_condition","delay_by_weather.png"),("carrier","delay_by_carrier.png"),("route_type","delay_by_route_type.png"),("state","delay_by_state.png")]:
            if col in df.columns:
                rates = df.groupby(col)[target].mean().sort_values(ascending=False).head(20)
                plt.figure(figsize=(9, 5)); rates.plot(kind="bar"); plt.title(f"Delay Rate by {col}"); plt.ylabel("Delay rate"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(Path(CONFIG.output_dir)/fname, dpi=150); plt.close()
        numeric = df.select_dtypes(include="number")
        if target in numeric.columns:
            corr = numeric.corr(numeric_only=True)[target].drop(target, errors="ignore").sort_values(key=lambda s: s.abs(), ascending=False)
            plt.figure(figsize=(10, 6)); corr.head(20).sort_values().plot(kind="barh"); plt.title("Top Numeric Correlations with Delay Risk"); plt.tight_layout(); plt.savefig(Path(CONFIG.output_dir)/"target_correlations.png", dpi=150); plt.close()
        drop_cols = [c for c in ["delivery_status","actual_delivery_days","delay_probability_true","order_id","batch_id","customer_id","customer_name","customer_email","address","order_date","ingestion_timestamp","order_timestamp"] if c in df.columns]
        ml_df = df.drop(columns=drop_cols, errors="ignore")
        cat_cols = ml_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        high_card = [c for c in cat_cols if c != target and ml_df[c].nunique(dropna=True) > CONFIG.max_cat_cardinality]
        ml_df = ml_df.drop(columns=high_card, errors="ignore")
        all_null = [c for c in ml_df.columns if c != target and ml_df[c].isna().all()]
        constants = [c for c in ml_df.columns if c != target and ml_df[c].nunique(dropna=True) <= 1]
        ml_df = ml_df.drop(columns=all_null + constants, errors="ignore")
        ml_df.to_parquet(P["ml_ready"], index=False)
        out = f"EDA_SUCCESS\nML-ready rows: {len(ml_df):,}\nML-ready columns: {len(ml_df.columns):,}\nDropped leakage/text/date cols: {drop_cols}\nDropped high-cardinality cols: {high_card}\nFile: {P['ml_ready']}"
        append_result("eda", out)
        return out
    except Exception as e:
        return f"EDA_ERROR: {e}\n{traceback.format_exc()}"


@tool("validate_business_hypotheses")
def validate_business_hypotheses(_: str = "") -> str:
    """Validates business hypotheses about delivery delay drivers."""
    try:
        df = pd.read_parquet(P["features"])
        target = CONFIG.target_col
        overall = float(df[target].mean())
        hyps = [("H1","Severe traffic increases delay risk.","traffic_condition","severe"),("H2","Storm weather increases delay risk.","weather_condition","storm"),("H3","Remote routes increase delay risk.","route_type","remote"),("H4","NationalPost has higher delay risk than average.","carrier","NationalPost"),("H5","Very long distance orders have higher delay risk.","distance_band","very_long"),("H6","Weekend orders have higher delay risk.","is_weekend",1),("H7","Peak-hour orders have higher delay risk.","is_peak_hour",1),("H8","Orders estimated above promised SLA have higher delay risk.","estimated_minus_promised_days","positive"),("H9","Very heavy orders have higher delay risk.","weight_band","very_heavy"),("H10","Different states have materially different delay rates.","state","top_vs_overall")]
        rows=[]
        for hid, stmt, col, cond in hyps:
            if col not in df.columns:
                rows.append({"id":hid,"hypothesis":stmt,"feature":col,"verdict":"INCONCLUSIVE","business_insight":f"Column {col} not found."}); continue
            if cond == "positive": mask = df[col] > 0
            elif cond == "top_vs_overall":
                rates = df.groupby(col)[target].agg(["mean","count"]); rates = rates[rates["count"] >= max(100, len(df)*0.001)]
                cond = str(rates["mean"].idxmax()); mask = df[col].astype(str) == cond
            else: mask = df[col].astype(str) == str(cond)
            seg_rate = float(df.loc[mask, target].mean()) if mask.sum() else np.nan
            lift = float(seg_rate / overall) if overall and mask.sum() else np.nan
            verdict = "TRUE" if mask.sum() >= 50 and lift >= 1.15 and seg_rate > overall else "FALSE" if mask.sum() >= 50 else "INCONCLUSIVE"
            rows.append({"id":hid,"hypothesis":stmt,"feature":col,"condition":cond,"segment_count":int(mask.sum()),"segment_rate":seg_rate,"overall_rate":overall,"lift":lift,"verdict":verdict,"business_insight":f"Orders matching {col}={cond} have {lift:.2f}x the average delay rate." if not pd.isna(lift) else "Insufficient sample."})
        save_json(P["hyp_json"], rows)
        outdf = pd.DataFrame(rows)
        Path(P["hyp_md"]).write_text("# Business Hypothesis Validation\n\n" + f"Overall delay rate: **{overall:.2%}**\n\n" + md_table(outdf, 20), encoding="utf-8")
        plot = outdf.dropna(subset=["lift"])
        plt.figure(figsize=(10,5)); plt.bar(plot["id"], plot["lift"]); plt.axhline(1.0, linestyle="--"); plt.title("Hypothesis Lift vs Overall Delay Rate"); plt.ylabel("Lift"); plt.tight_layout(); plt.savefig(P["hyp_png"], dpi=150); plt.close()
        out = f"HYPOTHESIS_SUCCESS\nHypotheses tested: {len(rows)}\nTRUE: {(outdf['verdict']=='TRUE').sum()}\nFile: {P['hyp_md']}"
        append_result("hypothesis", out)
        return out
    except Exception as e:
        return f"HYPOTHESIS_ERROR: {e}\n{traceback.format_exc()}"


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    best = {"threshold":0.5,"business_score":-1e18,"precision":0,"recall":0,"f1":0}
    for t in thresholds:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        gain = tp*CONFIG.value_true_positive - fp*CONFIG.cost_false_positive - fn*CONFIG.cost_false_negative
        score = gain / len(y_true)
        prec = precision_score(y_true, pred, zero_division=0); rec = recall_score(y_true, pred, zero_division=0); f1 = f1_score(y_true, pred, zero_division=0)
        if prec < 0.20: score -= 50
        if score > best["business_score"]: best = {"threshold":float(t),"business_score":float(score),"precision":float(prec),"recall":float(rec),"f1":float(f1)}
    return best["threshold"], best


def eval_model(name: str, model, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else np.asarray(model.predict(X_test)).astype(float)
    th, th_info = choose_threshold(y_test.values, prob)
    pred = (prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"name":name,"model":model,"threshold":th,"threshold_info":th_info,"roc_auc":float(roc_auc_score(y_test, prob)),"average_precision":float(average_precision_score(y_test, prob)),"accuracy":float(accuracy_score(y_test, pred)),"precision":float(precision_score(y_test, pred, zero_division=0)),"recall":float(recall_score(y_test, pred, zero_division=0)),"f1":float(f1_score(y_test, pred, zero_division=0)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"prob":prob,"pred":pred}


@tool("train_delay_risk_model")
def train_delay_risk_model(_: str = "") -> str:
    """Trains model candidates and saves the best delay-risk model artifact."""
    try:
        df = pd.read_parquet(P["ml_ready"])
        target = CONFIG.target_col
        y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
        X = df.drop(columns=[target])
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in X.select_dtypes(include=["object","string","category","bool"]).columns if X[c].nunique(dropna=True) <= CONFIG.max_cat_cardinality]
        feature_cols = num_cols + cat_cols
        X = X[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG.test_size, random_state=CONFIG.random_state, stratify=y)
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
        prep = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=CONFIG.n_jobs),
            "RandomForest": RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=CONFIG.n_jobs, random_state=CONFIG.random_state),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=150, max_depth=18, min_samples_leaf=15, class_weight="balanced", n_jobs=CONFIG.n_jobs, random_state=CONFIG.random_state),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=180, learning_rate=0.08, max_leaf_nodes=31, random_state=CONFIG.random_state),
        }
        if XGBClassifier is not None and CONFIG.enable_xgboost:
            models["XGBoost"] = XGBClassifier(n_estimators=220, max_depth=6, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", n_jobs=CONFIG.n_jobs, random_state=CONFIG.random_state)
        if LGBMClassifier is not None and CONFIG.enable_lightgbm:
            models["LightGBM"] = LGBMClassifier(n_estimators=240, learning_rate=0.06, num_leaves=48, class_weight="balanced", n_jobs=CONFIG.n_jobs, random_state=CONFIG.random_state, verbose=-1)
        results=[]; best=None
        for name, est in models.items():
            logger.info("[ML] Training %s", name)
            pipe = Pipeline([("preprocessor", prep), ("model", est)])
            try:
                res = eval_model(name, pipe, X_train, X_test, y_train, y_test)
                res["selection_score"] = float(0.40*res["recall"] + 0.25*res["roc_auc"] + 0.20*res["average_precision"] + 0.15*res["precision"])
                results.append(res)
                if best is None or res["selection_score"] > best["selection_score"]: best = res
                logger.info("[ML] %s AUC=%.4f AP=%.4f Precision=%.4f Recall=%.4f F1=%.4f Threshold=%.3f", name, res["roc_auc"], res["average_precision"], res["precision"], res["recall"], res["f1"], res["threshold"])
            except Exception as ex:
                logger.warning("[ML] %s failed: %s", name, ex)
        if best is None: return "ML_ERROR: no model trained successfully."
        X_out = X_test.copy(); X_out[target] = y_test.values; X_out["delay_probability_pred"] = best["prob"]; X_out["delay_prediction"] = best["pred"]; X_out.to_parquet(P["predictions"], index=False)
        fi=[]
        if permutation_importance is not None:
            try:
                sample_n = min(CONFIG.sample_for_importance, len(X_test)); Xp = X_test.sample(sample_n, random_state=CONFIG.random_state); yp = y_test.loc[Xp.index]
                perm = permutation_importance(best["model"], Xp, yp, n_repeats=5, random_state=CONFIG.random_state, scoring="average_precision", n_jobs=CONFIG.n_jobs)
                fi = [{"feature": feature_cols[i], "importance_mean": float(perm.importances_mean[i]), "importance_std": float(perm.importances_std[i])} for i in np.argsort(perm.importances_mean)[::-1][:30]]
            except Exception as ex: logger.warning("Feature importance failed: %s", ex)
        if fi:
            fdf = pd.DataFrame(fi).sort_values("importance_mean", ascending=True)
            plt.figure(figsize=(9,7)); plt.barh(fdf["feature"].tail(20), fdf["importance_mean"].tail(20)); plt.title(f"Top Feature Importance — {best['name']}"); plt.tight_layout(); plt.savefig(P["feature_importance_png"], dpi=150); plt.close()
        public=[]
        for r in results:
            public.append({k:r[k] for k in ["name","roc_auc","average_precision","accuracy","precision","recall","f1","threshold","selection_score","tn","fp","fn","tp"]} | {"business_score_per_order": r["threshold_info"]["business_score"]})
        metrics={"target":target,"best_model":best["name"],"best_threshold":best["threshold"],"features":feature_cols,"numeric_features":num_cols,"categorical_features":cat_cols,"results":public,"feature_importance":fi,"train_rows":int(len(X_train)),"test_rows":int(len(X_test))}
        save_json(P["metrics_json"], metrics)
        artifact={"model":best["model"],"target":target,"model_name":best["name"],"threshold":best["threshold"],"features":feature_cols,"numeric_features":num_cols,"categorical_features":cat_cols,"metrics":metrics,"created_by":"CommerceFlow AI CrewAI DS Pipeline"}
        with open(P["model"], "wb") as f: pickle.dump(artifact, f)
        mdf = pd.DataFrame(public).sort_values("selection_score", ascending=False)
        Path(P["metrics_md"]).write_text("# Model Metrics — Delay Risk Prediction\n\n" + f"Best model: **{best['name']}**  \nBest threshold: **{best['threshold']:.3f}**  \nTrain rows: **{len(X_train):,}**  \nTest rows: **{len(X_test):,}**\n\n" + md_table(mdf, 20) + "\n\n## Classification Report\n```text\n" + classification_report(y_test, best["pred"], zero_division=0) + "\n```", encoding="utf-8")
        Path(P["error_md"]).write_text("# Error Analysis\n\nFalse negatives are the most critical error because they represent delayed orders that were not flagged.\n\n" + f"FN: **{best['fn']:,}**  \nFP: **{best['fp']:,}**\n", encoding="utf-8")
        out = f"ML_SUCCESS\nBest model: {best['name']}\nROC AUC: {best['roc_auc']:.4f}\nAverage Precision: {best['average_precision']:.4f}\nPrecision: {best['precision']:.4f}\nRecall: {best['recall']:.4f}\nF1: {best['f1']:.4f}\nThreshold: {best['threshold']:.3f}\nModel artifact: {P['model']}\nPredictions: {P['predictions']}"
        append_result("ml", out)
        return out
    except Exception as e:
        return f"ML_ERROR: {e}\n{traceback.format_exc()}"


@tool("evaluate_business_performance")
def evaluate_business_performance(_: str = "") -> str:
    """Converts model results into business performance and operational value."""
    try:
        metrics = load_json(P["metrics_json"], {})
        preds = pd.read_parquet(P["predictions"])
        y = preds[CONFIG.target_col].astype(int).values; p = preds["delay_prediction"].astype(int).values
        tn, fp, fn, tp = confusion_matrix(y, p).ravel()
        model_value = tp*CONFIG.value_true_positive - fp*CONFIG.cost_false_positive - fn*CONFIG.cost_false_negative
        baseline_value = -int(y.sum())*CONFIG.cost_false_negative
        incremental = model_value - baseline_value
        per_1000 = incremental / len(y) * 1000
        perf = pd.DataFrame([{"metric":"true_positives","value":tp},{"metric":"false_positives","value":fp},{"metric":"false_negatives","value":fn},{"metric":"true_negatives","value":tn},{"metric":"model_value","value":model_value},{"metric":"baseline_value","value":baseline_value},{"metric":"incremental_value","value":incremental},{"metric":"incremental_value_per_1000_orders","value":per_1000}])
        plt.figure(figsize=(8,5)); plt.bar(["Baseline", "Model"], [baseline_value, model_value]); plt.title("Estimated Business Value — Baseline vs Model"); plt.ylabel("Value units"); plt.tight_layout(); plt.savefig(P["business_png"], dpi=150); plt.close()
        md = "# Business Performance — Delay Risk Model\n\n" + f"Best model: **{metrics.get('best_model', 'N/A')}**  \nThreshold: **{metrics.get('best_threshold', 'N/A')}**\n\n" + "## Cost assumptions\n" + f"- False negative cost: {CONFIG.cost_false_negative}\n- False positive handling cost: {CONFIG.cost_false_positive}\n- True positive value captured: {CONFIG.value_true_positive}\n\n" + "## Estimated Value\n" + md_table(perf, 20) + "\n\n" + f"The model improves the simulated business value by **{incremental:,.2f}**, equivalent to **{per_1000:,.2f} per 1,000 orders**."
        Path(P["business_md"]).write_text(md, encoding="utf-8")
        out = f"BUSINESS_PERFORMANCE_SUCCESS\nIncremental value: {incremental:,.2f}\nValue per 1,000 orders: {per_1000:,.2f}\nFile: {P['business_md']}"
        append_result("business", out)
        return out
    except Exception as e:
        return f"BUSINESS_PERFORMANCE_ERROR: {e}\n{traceback.format_exc()}"


@tool("generate_analysis_notebook")
def generate_analysis_notebook(_: str = "") -> str:
    """Generates a detailed technical notebook including source code and artifacts."""
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

        manifest_path = write_project_code_manifest()
        code_files = load_json(manifest_path, [])

        cells = []
        cells.append(new_markdown_cell(f'''# {CONFIG.project_name}

Technical notebook generated by the CommerceFlow AI CrewAI DS pipeline.

This notebook is designed to be a portfolio artifact and an audit notebook. It includes the key outputs:

- `ml_outputs/model.pkl`
- `ml_outputs/model_metrics.json`
- `ml_outputs/Model_Metrics.md`
- `ml_outputs/df4_predictions.parquet`
- `ml_outputs/Business_Performance.md`
- `ml_outputs/analysis_notebook.ipynb`
- `ml_outputs/dashboard.html`

It also includes a **Full Source Code Appendix** with the source files from the project.
'''))

        cells.append(new_markdown_cell("## 0. Environment Setup"))
        cells.append(new_code_cell('''from pathlib import Path
import json
import pickle
import pandas as pd
from IPython.display import Markdown, Image, display

OUT = Path("ml_outputs")
print("Output folder:", OUT.resolve())
'''))

        sections = [
            ("1. Pipeline Run Results", '''p = OUT / "pipeline_run_results.json"
if p.exists():
    display(pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))))
else:
    print("pipeline_run_results.json not found.")'''),
            ("2. Silver Orders", '''p = OUT / "df1_silver_orders.parquet"
if p.exists():
    df = pd.read_parquet(p)
    print("Shape:", df.shape)
    display(df.head())
    display(df.dtypes.to_frame("dtype"))
else:
    print("df1_silver_orders.parquet not found.")'''),
            ("3. Data Quality Report", '''p = OUT / "Data_Quality_Report.md"
if p.exists():
    display(Markdown(p.read_text(encoding="utf-8")))
else:
    print("Data_Quality_Report.md not found.")'''),
            ("4. Feature Engineering Output", '''p = OUT / "df2_features.parquet"
if p.exists():
    df = pd.read_parquet(p)
    print("Shape:", df.shape)
    display(df.head())
    display(df.describe(include="all").T.head(80))
else:
    print("df2_features.parquet not found.")'''),
            ("5. ML-ready Dataset", '''p = OUT / "df3_ml_ready.parquet"
if p.exists():
    df = pd.read_parquet(p)
    print("Shape:", df.shape)
    display(df.head())
    if "delay_risk_label" in df.columns:
        display(df["delay_risk_label"].value_counts(normalize=True).to_frame("share"))
else:
    print("df3_ml_ready.parquet not found.")'''),
            ("6. EDA Visuals", '''for img in [
    "target_distribution.png", "delay_by_traffic.png", "delay_by_weather.png",
    "delay_by_carrier.png", "delay_by_route_type.png", "delay_by_state.png",
    "target_correlations.png",
]:
    p = OUT / img
    if p.exists():
        print("\\n" + img)
        display(Image(filename=str(p)))
    else:
        print("Missing:", img)'''),
            ("7. Hypothesis Validation", '''p = OUT / "Hypothesis_Validation.md"
if p.exists():
    display(Markdown(p.read_text(encoding="utf-8")))
else:
    print("Hypothesis_Validation.md not found.")

p = OUT / "hypothesis_results.json"
if p.exists():
    display(pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))))'''),
            ("8. Model Metrics and Competition", '''p = OUT / "Model_Metrics.md"
if p.exists():
    display(Markdown(p.read_text(encoding="utf-8")))
else:
    print("Model_Metrics.md not found.")

p = OUT / "model_metrics.json"
if p.exists():
    metrics = json.loads(p.read_text(encoding="utf-8"))
    print("Best model:", metrics.get("best_model"))
    print("Best threshold:", metrics.get("best_threshold"))
    display(pd.DataFrame(metrics.get("results", [])).sort_values("selection_score", ascending=False))
else:
    print("model_metrics.json not found.")'''),
            ("9. Feature Importance", '''p = OUT / "feature_importance.png"
if p.exists():
    display(Image(filename=str(p)))
else:
    print("feature_importance.png not found.")'''),
            ("10. Model Artifact Inspection", '''p = OUT / "model.pkl"
if p.exists():
    with open(p, "rb") as f:
        artifact = pickle.load(f)
    print("Artifact keys:", artifact.keys())
    print("Model name:", artifact.get("model_name"))
    print("Target:", artifact.get("target"))
    print("Threshold:", artifact.get("threshold"))
    print("Feature count:", len(artifact.get("features", [])))
    print("Numeric features:", artifact.get("numeric_features", [])[:20])
    print("Categorical features:", artifact.get("categorical_features", [])[:20])
else:
    print("model.pkl not found.")'''),
            ("11. Predictions Output", '''p = OUT / "df4_predictions.parquet"
if p.exists():
    preds = pd.read_parquet(p)
    print("Shape:", preds.shape)
    display(preds.head())
    cols = ["delay_risk_label", "delay_probability_pred", "delay_prediction"]
    if set(cols).issubset(preds.columns):
        display(preds[cols].describe())
else:
    print("df4_predictions.parquet not found.")'''),
            ("12. Error Analysis", '''p = OUT / "Error_Analysis.md"
if p.exists():
    display(Markdown(p.read_text(encoding="utf-8")))
else:
    print("Error_Analysis.md not found.")'''),
            ("13. Business Performance", '''p = OUT / "Business_Performance.md"
if p.exists():
    display(Markdown(p.read_text(encoding="utf-8")))
else:
    print("Business_Performance.md not found.")

p = OUT / "business_performance.png"
if p.exists():
    display(Image(filename=str(p)))'''),
            ("14. Static HTML Dashboard", '''p = OUT / "dashboard.html"
if p.exists():
    print("HTML dashboard:", p.resolve())
else:
    print("dashboard.html not found.")'''),
        ]

        for title, code in sections:
            cells.append(new_markdown_cell(f"## {title}"))
            cells.append(new_code_cell(code))

        cells.append(new_markdown_cell('''## 15. Production Inference Pattern

The `model.pkl` file stores the full sklearn preprocessing + model pipeline, model name, target, threshold, feature list and metrics.

```python
with open("ml_outputs/model.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
threshold = artifact["threshold"]
prob = model.predict_proba(new_order_features)[0, 1]
risk = int(prob >= threshold)
```
'''))

        cells.append(new_markdown_cell("## 16. Full Source Code Appendix"))
        cells.append(new_code_cell(f'''manifest_path = Path(r"{manifest_path}")
if manifest_path.exists():
    code_files = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Files included in source appendix: {{len(code_files)}}")
    display(pd.DataFrame([{{"path": f["path"], "language": f["language"], "chars": len(f["content"])}} for f in code_files]))
else:
    print("Project code manifest not found.")
'''))

        for fobj in code_files:
            path = fobj.get("path", "")
            content = fobj.get("content", "")
            cells.append(new_markdown_cell(f"### Source file: `{path}`"))
            cells.append(new_code_cell(f"# File: {path}\\n" + content))

        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

        with open(P["notebook"], "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        out = f"NOTEBOOK_SUCCESS\\nFile: {P['notebook']}\\nSource-code manifest: {manifest_path}\\nSource files embedded: {len(code_files)}"
        append_result("notebook", out)
        return out

    except Exception as e:
        return f"NOTEBOOK_ERROR: {e}\\n{traceback.format_exc()}"


@tool("generate_html_dashboard")
def generate_html_dashboard(_: str = "") -> str:
    """Generates a static HTML dashboard with model and business outputs."""
    try:
        metrics = load_json(P["metrics_json"], {}); hyps = load_json(P["hyp_json"], [])
        res = pd.DataFrame(metrics.get("results", [])); best_row = res.sort_values("selection_score", ascending=False).iloc[0].to_dict() if not res.empty else {}
        model_rows = "" if res.empty else "".join([f"<tr><td>{r['name']}</td><td>{r['roc_auc']:.4f}</td><td>{r['average_precision']:.4f}</td><td>{r['precision']:.4f}</td><td>{r['recall']:.4f}</td><td>{r['f1']:.4f}</td></tr>" for _, r in res.sort_values("selection_score", ascending=False).iterrows()])
        hyp_rows = "".join([f"<tr><td>{h.get('id')}</td><td>{h.get('hypothesis')}</td><td>{h.get('verdict')}</td><td>{h.get('lift','')}</td></tr>" for h in hyps])
        business = Path(P["business_md"]).read_text(encoding="utf-8") if Path(P["business_md"]).exists() else ""
        html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>CommerceFlow AI — Delay Risk</title><style>body{{margin:0;font-family:Inter,Segoe UI,Arial;background:#0f172a;color:#e5e7eb}}.container{{max-width:1150px;margin:auto;padding:36px 20px}}.hero,.card{{border:1px solid #334155;border-radius:18px;background:#111827;padding:24px;margin:18px 0}}h1{{font-size:42px;margin:0}}p{{color:#cbd5e1}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.metric{{border:1px solid #334155;border-radius:14px;padding:18px;background:#020617}}.metric span{{color:#94a3b8}}.metric strong{{display:block;font-size:26px;margin-top:8px}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #334155;padding:10px;text-align:left}}th{{color:#38bdf8}}img{{max-width:100%;background:white;border-radius:12px}}pre{{white-space:pre-wrap;background:#020617;border:1px solid #334155;border-radius:12px;padding:16px}}</style></head><body><div class="container"><section class="hero"><p>Real-time e-commerce · ML production simulation</p><h1>CommerceFlow AI — Delay Risk Model</h1><p>End-to-end CrewAI data science pipeline for delivery delay risk prediction.</p></section><section class="grid"><div class="metric"><span>Best Model</span><strong>{metrics.get('best_model','N/A')}</strong></div><div class="metric"><span>ROC AUC</span><strong>{best_row.get('roc_auc',0):.3f}</strong></div><div class="metric"><span>Precision</span><strong>{best_row.get('precision',0):.3f}</strong></div><div class="metric"><span>Recall</span><strong>{best_row.get('recall',0):.3f}</strong></div></section><section class="card"><h2>Model Competition</h2><table><thead><tr><th>Model</th><th>ROC AUC</th><th>AP</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{model_rows}</tbody></table></section><section class="card"><h2>Hypothesis Validation</h2><table><thead><tr><th>ID</th><th>Hypothesis</th><th>Verdict</th><th>Lift</th></tr></thead><tbody>{hyp_rows}</tbody></table></section><section class="card"><h2>Feature Importance</h2><img src="feature_importance.png"></section><section class="card"><h2>Business Performance</h2><img src="business_performance.png"><pre>{business[:3500]}</pre></section></div></body></html>'''
        Path(P["html"]).write_text(html, encoding="utf-8")
        out=f"HTML_SUCCESS\nFile: {P['html']}"; append_result("html", out); return out
    except Exception as e:
        return f"HTML_ERROR: {e}\n{traceback.format_exc()}"


@tool("gitops_commit_and_push")
def gitops_commit_and_push(_: str = "") -> str:
    """Commits and pushes project code and lightweight generated outputs to GitHub."""
    try:
        import subprocess

        if not CONFIG.git_enabled:
            out = "GITOPS_SKIPPED: CONFIG.git_enabled=False"
            append_result("gitops", out)
            return out

        root = Path(".").resolve()
        if not (root / ".git").exists():
            out = (
                "GITOPS_SKIPPED: this folder is not a git repository yet.\\n"
                "Run:\\n"
                "git init\\n"
                "git branch -M main\\n"
                "git remote add origin https://github.com/bttisrael/ecommerceflow.git"
            )
            append_result("gitops", out)
            return out

        gitignore_path = root / ".gitignore"
        ignore_lines = []
        if gitignore_path.exists():
            ignore_lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()

        additions = [".venv/", "venv/", "__pycache__/", "node_modules/", "*.log", "*.db", "*.sqlite", "*.sqlite3"]
        if not CONFIG.git_track_large_artifacts:
            additions += ["*.parquet", "*.pkl"]

        changed = False
        for item in additions:
            if item not in ignore_lines:
                ignore_lines.append(item)
                changed = True
        if changed:
            gitignore_path.write_text("\\n".join(ignore_lines) + "\\n", encoding="utf-8")

        def run(cmd: str):
            return subprocess.run(cmd, cwd=str(root), shell=True, text=True, capture_output=True)

        for item in ["commerceflow_delay_risk_crewai.py", "generate_historical_orders.py", "README.md", "requirements.txt", ".gitignore", "backend", "frontend", "dashboard", "sql", "ml"]:
            if Path(item).exists():
                run(f'git add "{item}"')

        if Path(CONFIG.output_dir).exists():
            for pattern in [f"{CONFIG.output_dir}/*.md", f"{CONFIG.output_dir}/*.json", f"{CONFIG.output_dir}/*.html", f"{CONFIG.output_dir}/*.ipynb", f"{CONFIG.output_dir}/*.png"]:
                run(f"git add {pattern}")
            if CONFIG.git_track_large_artifacts:
                for pattern in [f"{CONFIG.output_dir}/*.pkl", f"{CONFIG.output_dir}/*.parquet"]:
                    run(f"git add -f {pattern}")

        status = run("git status --short").stdout.strip()
        if not status:
            out = "GITOPS_SUCCESS: no changes to commit."
            append_result("gitops", out)
            return out

        msg = f'{CONFIG.git_commit_prefix} ({pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")})'
        commit = run(f'git commit -m "{msg}"')
        if commit.returncode != 0 and "nothing to commit" not in (commit.stdout + commit.stderr).lower():
            out = f"GITOPS_ERROR: commit failed\\nSTDOUT:\\n{commit.stdout}\\nSTDERR:\\n{commit.stderr}"
            append_result("gitops", out)
            return out

        if not CONFIG.git_push:
            out = "GITOPS_SUCCESS: committed locally. Push skipped by CONFIG.git_push=False."
            append_result("gitops", out)
            return out

        push = run(f"git push {CONFIG.git_remote_name} {CONFIG.git_branch}")
        if push.returncode != 0 and "fetch first" in (push.stdout + push.stderr).lower():
            pull = run(f"git pull --rebase {CONFIG.git_remote_name} {CONFIG.git_branch}")
            if pull.returncode != 0:
                out = f"GITOPS_ERROR: pull --rebase failed\\nSTDOUT:\\n{pull.stdout}\\nSTDERR:\\n{pull.stderr}"
                append_result("gitops", out)
                return out
            push = run(f"git push {CONFIG.git_remote_name} {CONFIG.git_branch}")

        if push.returncode != 0:
            out = f"GITOPS_ERROR: push failed\\nSTDOUT:\\n{push.stdout}\\nSTDERR:\\n{push.stderr}"
            append_result("gitops", out)
            return out

        out = (
            f"GITOPS_SUCCESS\\nCommitted and pushed to {CONFIG.git_remote_name}/{CONFIG.git_branch}.\\n"
            f"Tracked large artifacts: {CONFIG.git_track_large_artifacts}\\n"
            "model.pkl and parquet files are ignored unless --track-large-artifacts is used."
        )
        append_result("gitops", out)
        return out

    except Exception as e:
        return f"GITOPS_ERROR: {e}\\n{traceback.format_exc()}"


DIRECT_STEPS = [("Ingestion", ingest_orders_from_sqlite),("Data Analysis", analyze_data_quality),("Feature Engineering", build_features),("EDA", generate_eda_and_ml_ready),("Hypothesis", validate_business_hypotheses),("ML", train_delay_risk_model),("Business Performance", evaluate_business_performance),("Notebook", generate_analysis_notebook),("HTML", generate_html_dashboard),("GitOps", gitops_commit_and_push)]



def _tool_name(fn) -> str:
    """
    CrewAI @tool converts normal functions into Tool objects.
    Tool objects do not have __name__, so CrewAI mode must read .name instead.
    """
    return getattr(fn, "__name__", None) or getattr(fn, "name", None) or fn.__class__.__name__


def build_crew():
    if not CREWAI_AVAILABLE: raise RuntimeError("CrewAI is not installed. Use --direct or install crewai.")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY not found. Use --direct or configure .env.")
    llm = LLM(model=CONFIG.anthropic_model, api_key=api_key, temperature=0.0)
    specs=[("CommerceFlow Data Ingestor","Load historical orders from SQLite.",ingest_orders_from_sqlite), ("Data Quality Analyst","Analyze quality, target and leakage.",analyze_data_quality), ("Feature Engineer","Build logistics delay-risk features.",build_features), ("EDA Analyst","Create charts and ML-ready table.",generate_eda_and_ml_ready), ("Hypothesis Validator","Validate business hypotheses.",validate_business_hypotheses), ("ML Scientist","Train and save best model.pkl.",train_delay_risk_model), ("Business Performance Analyst","Translate metrics into value.",evaluate_business_performance), ("Notebook Writer","Generate technical notebook.",generate_analysis_notebook), ("HTML Dashboard Writer","Generate stakeholder dashboard.",generate_html_dashboard), ("GitOps Agent","Commit and push code and outputs to GitHub.",gitops_commit_and_push)]
    agents=[]; tasks=[]; prev=None
    for role, goal, fn in specs:
        ag=Agent(role=role, goal=goal, backstory=f"Senior specialist responsible for: {goal}", tools=[fn], llm=llm, verbose=True, max_iter=4, max_retry_limit=2)
        tool_name = _tool_name(fn)
        task=Task(description=f"Call {tool_name}. Finish only when SUCCESS is returned.", agent=ag, context=[prev] if prev else None, expected_output=f"{tool_name} SUCCESS output.")
        agents.append(ag); tasks.append(task); prev=task
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, memory=False, verbose=True)


def _call_tool_or_function(fn, arg: str = "") -> str:
    """
    CrewAI's @tool decorator converts functions into Tool objects.
    In direct mode, Tool objects are not callable, so we call .run().
    Plain functions are still called normally.
    """
    if callable(fn):
        return fn(arg)
    if hasattr(fn, "run"):
        return fn.run(arg)
    if hasattr(fn, "_run"):
        return fn._run(arg)
    raise TypeError(f"Unsupported step object: {type(fn)}")


def run_direct():
    for name, fn in DIRECT_STEPS:
        print("\n" + "="*80 + f"\n{name}\n" + "="*80)
        res = _call_tool_or_function(fn, "")
        print(res)
        if "_ERROR" in str(res):
            raise RuntimeError(str(res))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--db-path", default=CONFIG.db_path)
    p.add_argument("--table-name", default=CONFIG.table_name)
    p.add_argument("--max-rows", type=int, default=CONFIG.max_rows)
    p.add_argument("--target", default=CONFIG.target_col)
    p.add_argument("--output-dir", default=CONFIG.output_dir)
    p.add_argument("--n-jobs", type=int, default=CONFIG.n_jobs)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--no-xgboost", action="store_true")
    p.add_argument("--no-lightgbm", action="store_true")
    p.add_argument("--no-git", action="store_true", help="Disable GitOps commit/push step.")
    p.add_argument("--no-git-push", action="store_true", help="Commit locally but do not push.")
    p.add_argument("--track-large-artifacts", action="store_true", help="Force GitOps to track model.pkl/parquet artifacts.")
    p.add_argument("--git-remote", default=CONFIG.git_remote_name)
    p.add_argument("--git-branch", default=CONFIG.git_branch)
    return p.parse_args()


def apply_args(a):
    CONFIG.db_path=a.db_path; CONFIG.table_name=a.table_name; CONFIG.max_rows=a.max_rows; CONFIG.target_col=a.target; CONFIG.output_dir=a.output_dir; CONFIG.n_jobs=a.n_jobs; CONFIG.enable_xgboost=not a.no_xgboost; CONFIG.enable_lightgbm=not a.no_lightgbm; CONFIG.git_enabled=not a.no_git; CONFIG.git_push=not a.no_git_push; CONFIG.git_track_large_artifacts=a.track_large_artifacts; CONFIG.git_remote_name=a.git_remote; CONFIG.git_branch=a.git_branch; refresh_paths()


def main():
    args=parse_args(); apply_args(args); ensure_out(); save_json(P["run_results"], [])
    logger.info("Starting CommerceFlow AI DS pipeline | DB=%s | max_rows=%s", CONFIG.db_path, CONFIG.max_rows)
    if args.direct:
        run_direct()
    else:
        crew = build_crew(); crew.kickoff()
    print("\nPIPELINE COMPLETED")
    print(f"Model artifact: {P['model']}")
    print(f"Metrics:        {P['metrics_md']}")
    print(f"Dashboard:      {P['html']}")
    print(f"Notebook:       {P['notebook']}")
    print(f"GitOps:         {'enabled' if CONFIG.git_enabled else 'disabled'}")


if __name__ == "__main__":
    main()
