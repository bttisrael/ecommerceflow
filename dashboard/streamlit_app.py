from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "delay_risk_model.pkl"

st.set_page_config(page_title="CommerceFlow AI Dashboard", layout="wide")
st.title("CommerceFlow AI — Order Monitoring & Delay Risk")
st.caption("Reads local simulated orders. Later this can read BigQuery and call Vertex AI.")

from backend.database import export_orders_dataframe

df = export_orders_dataframe()
if df.empty:
    st.warning("No orders found. Run the FastAPI backend and generate orders first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders", f"{len(df):,}")
c2.metric("Revenue", f"R$ {df['order_value'].sum():,.0f}")
c3.metric("Avg Distance", f"{df['distance_km'].mean():,.1f} km")
c4.metric("Delay Risk Rate", f"{df['delay_risk_label'].mean()*100:.1f}%")

if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    pred_df = df.copy()
    pred_df["order_timestamp"] = pd.to_datetime(pred_df["order_timestamp"], errors="coerce")
    pred_df["day_of_week"] = pred_df["order_timestamp"].dt.dayofweek.fillna(0).astype(int)
    pred_df["order_hour"] = pred_df["order_timestamp"].dt.hour.fillna(0).astype(int)
    pred_df["is_weekend"] = pred_df["day_of_week"].isin([5, 6]).astype(int)
    feature_cols = ["state", "city", "product_category", "order_value", "order_weight_kg", "distance_km", "carrier", "dc_id", "promised_delivery_days", "estimated_delivery_days", "day_of_week", "order_hour", "is_weekend"]
    df["ml_delay_risk_score"] = model.predict_proba(pred_df[feature_cols])[:, 1]
    st.success("ML model loaded. Showing predicted delay risk scores.")
else:
    df["ml_delay_risk_score"] = df["delay_risk_score_rule"]
    st.info("ML model not found yet. Showing rule-based risk score. Train with `python ml/train_delay_model.py`.")

left, right = st.columns(2)
with left:
    st.subheader("Orders by State")
    st.bar_chart(df.groupby("state")["order_id"].count().sort_values(ascending=False))
with right:
    st.subheader("Delay Risk by Carrier")
    st.bar_chart(df.groupby("carrier")["delay_risk_label"].mean().sort_values(ascending=False))

st.subheader("High Risk Orders")
st.dataframe(df.sort_values("ml_delay_risk_score", ascending=False).head(100), width="stretch")
