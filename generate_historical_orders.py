"""
CommerceFlow AI — Historical Order Generator
============================================

Generates a large synthetic historical e-commerce/logistics dataset for delay-risk modeling.

Default:
- 2,000,000 historical orders
- Daily volume between 1,000 and 10,000 orders/day
- Starts on 2025-06-01
- Writes to SQLite table `orders`
- Optional Parquet chunks for ML/BigQuery loading

Usage:
    python generate_historical_orders.py --rows 2000000 --db-path commerceflow.db

Safer first test:
    python generate_historical_orders.py --rows 100000 --db-path commerceflow.db

Optional parquet output:
    python generate_historical_orders.py --rows 2000000 --db-path commerceflow.db --parquet-dir data/historical_parquet
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


BRAZIL_CITIES = [
    ("São Paulo", "SP", -23.5505, -46.6333, "Southeast", 1.00),
    ("Rio de Janeiro", "RJ", -22.9068, -43.1729, "Southeast", 0.72),
    ("Belo Horizonte", "MG", -19.9167, -43.9345, "Southeast", 0.62),
    ("Curitiba", "PR", -25.4284, -49.2733, "South", 0.55),
    ("Porto Alegre", "RS", -30.0346, -51.2177, "South", 0.47),
    ("Florianópolis", "SC", -27.5949, -48.5482, "South", 0.38),
    ("Brasília", "DF", -15.7939, -47.8828, "Central-West", 0.43),
    ("Goiânia", "GO", -16.6869, -49.2648, "Central-West", 0.35),
    ("Salvador", "BA", -12.9777, -38.5016, "Northeast", 0.44),
    ("Recife", "PE", -8.0476, -34.8770, "Northeast", 0.39),
    ("Fortaleza", "CE", -3.7319, -38.5267, "Northeast", 0.36),
    ("Manaus", "AM", -3.1190, -60.0217, "North", 0.25),
    ("Belém", "PA", -1.4558, -48.4902, "North", 0.27),
    ("Cuiabá", "MT", -15.6014, -56.0979, "Central-West", 0.22),
    ("Campo Grande", "MS", -20.4697, -54.6201, "Central-West", 0.20),
    ("Vitória", "ES", -20.2976, -40.2958, "Southeast", 0.24),
    ("Ribeirão Preto", "SP", -21.1699, -47.8099, "Southeast", 0.29),
    ("Campinas", "SP", -22.9056, -47.0608, "Southeast", 0.34),
    ("Joinville", "SC", -26.3044, -48.8487, "South", 0.18),
    ("Londrina", "PR", -23.3045, -51.1696, "South", 0.16),
]

DCS = [
    ("DC_SP", "São Paulo", "SP", -23.5505, -46.6333, 1.00),
    ("DC_PR", "Curitiba", "PR", -25.4284, -49.2733, 0.70),
    ("DC_PE", "Recife", "PE", -8.0476, -34.8770, 0.55),
    ("DC_GO", "Goiânia", "GO", -16.6869, -49.2648, 0.45),
]

PRODUCTS = [
    ("P001", "Smartphone Orion X", "Electronics", 2499.90, 0.45, 0.06),
    ("P002", "Wireless Headphones", "Electronics", 399.90, 0.25, 0.04),
    ("P003", "Premium Coffee Pack", "Grocery", 89.90, 1.20, 0.02),
    ("P004", "Skin Care Kit", "Beauty", 279.90, 0.65, 0.03),
    ("P005", "Running Shoes", "Fashion", 349.90, 0.90, 0.03),
    ("P006", "Robot Vacuum", "Home", 1599.90, 4.80, 0.08),
    ("P007", "Office Chair", "Home", 899.90, 12.50, 0.12),
    ("P008", "Baby Care Bundle", "Baby", 219.90, 2.20, 0.04),
    ("P009", "Gaming Keyboard", "Electronics", 599.90, 0.85, 0.05),
    ("P010", "Protein Pack", "Health", 149.90, 1.80, 0.03),
    ("P011", "Air Fryer", "Home", 499.90, 5.10, 0.07),
    ("P012", "Smart Watch", "Electronics", 899.90, 0.30, 0.05),
]

CARRIERS = [
    ("BlueExpress", 0.055, 1.00),
    ("RoadRunner", 0.075, 0.95),
    ("EcoFrete", 0.105, 0.85),
    ("FlashLog", 0.090, 1.10),
    ("NationalPost", 0.130, 0.75),
]

FIRST_NAMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduardo", "Fernanda", "Gabriel", "Helena",
    "Igor", "Juliana", "Lucas", "Mariana", "Nicolas", "Olivia", "Paulo", "Rafaela",
    "Thiago", "Vanessa", "William", "Yasmin", "Emanuel", "Isis", "Antonio", "Ryan",
    "Zoe", "Luiza", "Mateus", "Camila", "Renato", "Bianca"
]

LAST_NAMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Costa", "Pereira", "Rodrigues", "Almeida",
    "Nascimento", "Lima", "Araujo", "Fernandes", "Carvalho", "Gomes", "Martins",
    "Rocha", "Ribeiro", "Mendes", "Barbosa", "Correia"
]


def haversine_np(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def weighted_choice(rng, weights, size):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return rng.choice(len(w), size=size, p=w)


def build_daily_counts(total_rows: int, start_date: str, min_daily: int, max_daily: int, seed: int):
    rng = np.random.default_rng(seed)
    dates = []
    counts = []
    remaining = total_rows
    day = pd.Timestamp(start_date)

    while remaining > 0:
        dow = day.dayofweek
        weekend_factor = 0.78 if dow >= 5 else 1.0
        month_factor = 1.25 if day.month in [11, 12] else 1.0
        monday_factor = 1.12 if dow == 0 else 1.0
        base = int(rng.integers(min_daily, max_daily + 1) * weekend_factor * month_factor * monday_factor)
        n = max(min_daily, min(max_daily, base))
        n = min(n, remaining)
        dates.append(day)
        counts.append(n)
        remaining -= n
        day += pd.Timedelta(days=1)

    return pd.DataFrame({"date": dates, "orders": counts})


def generate_datetimes_for_days(daily: pd.DataFrame, rng) -> pd.Series:
    pieces = []
    for row in daily.itertuples(index=False):
        n = int(row.orders)
        hours = np.arange(24)
        hour_weights = (
            0.15
            + 1.2 * np.exp(-((hours - 12) ** 2) / 14)
            + 1.6 * np.exp(-((hours - 20) ** 2) / 10)
            + 0.55 * np.exp(-((hours - 9) ** 2) / 8)
        )
        hour_weights[:6] *= 0.25
        hour_weights = hour_weights / hour_weights.sum()

        h = rng.choice(hours, size=n, p=hour_weights)
        m = rng.integers(0, 60, size=n)
        s = rng.integers(0, 60, size=n)
        base = pd.to_datetime(row.date)
        day_dt = base + pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m") + pd.to_timedelta(s, unit="s")
        pieces.append(pd.Series(day_dt))

    return pd.concat(pieces, ignore_index=True)


def create_orders_chunk(start_idx: int, n: int, start_date: str, daily_counts_slice: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed + start_idx)

    city_idx = weighted_choice(rng, [c[5] for c in BRAZIL_CITIES], n)
    product_idx = weighted_choice(rng, [1.0] * len(PRODUCTS), n)
    carrier_idx = weighted_choice(rng, [1.0, 0.9, 0.75, 0.85, 0.65], n)

    cities = np.array(BRAZIL_CITIES, dtype=object)[city_idx]
    products = np.array(PRODUCTS, dtype=object)[product_idx]
    carriers = np.array(CARRIERS, dtype=object)[carrier_idx]

    customer_lat = cities[:, 2].astype(float) + rng.normal(0, 0.13, n)
    customer_lon = cities[:, 3].astype(float) + rng.normal(0, 0.13, n)

    dc_arr = np.array(DCS, dtype=object)
    dc_distances = []
    for dc in DCS:
        dc_distances.append(haversine_np(customer_lat, customer_lon, float(dc[3]), float(dc[4])))
    dc_distances = np.vstack(dc_distances).T
    nearest = dc_distances.argmin(axis=1)

    overflow = rng.random(n) < 0.15
    random_dc = rng.integers(0, len(DCS), n)
    dc_idx = np.where(overflow, random_dc, nearest)
    dcs = dc_arr[dc_idx]

    dc_lat = dcs[:, 3].astype(float)
    dc_lon = dcs[:, 4].astype(float)
    distance_km = haversine_np(customer_lat, customer_lon, dc_lat, dc_lon)

    order_ts = generate_datetimes_for_days(daily_counts_slice, rng)
    if len(order_ts) != n:
        start = pd.Timestamp(start_date)
        order_ts = start + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, n), unit="s")

    dow = order_ts.dt.dayofweek.to_numpy()
    hour = order_ts.dt.hour.to_numpy()
    month = order_ts.dt.month.to_numpy()
    is_weekend = (dow >= 5).astype(int)
    is_peak_hour = np.isin(hour, [8, 9, 10, 11, 12, 18, 19, 20, 21]).astype(int)

    weather_choices = np.array(["clear", "rain", "storm", "fog", "heatwave"], dtype=object)
    weather_probs = np.array([0.62, 0.24, 0.06, 0.04, 0.04])
    weather = rng.choice(weather_choices, size=n, p=weather_probs)

    traffic = []
    traffic_choices = np.array(["low", "medium", "high", "severe"], dtype=object)
    for peak in is_peak_hour:
        probs = np.array([0.12, 0.38, 0.36, 0.14]) if peak else np.array([0.35, 0.43, 0.17, 0.05])
        traffic.append(rng.choice(traffic_choices, p=probs))
    traffic = np.array(traffic, dtype=object)

    route_type = rng.choice(np.array(["urban", "regional", "long_haul", "remote"], dtype=object), size=n, p=[0.38, 0.32, 0.22, 0.08])

    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)
    customer_name = np.char.add(np.char.add(first.astype(str), " "), last.astype(str))
    customer_id = np.array([f"CUST-{start_idx+i:09d}" for i in range(n)], dtype=object)
    customer_email = np.array([f"{first[i].lower()}.{last[i].lower()}.{start_idx+i}@example.com" for i in range(n)], dtype=object)

    product_base_price = products[:, 3].astype(float)
    price_noise = rng.normal(1.0, 0.12, n)
    seasonal_boost = np.where(np.isin(month, [11, 12]), 1.08, 1.0)
    order_value = np.maximum(19.9, product_base_price * price_noise * seasonal_boost).round(2)

    order_weight_kg = np.maximum(0.05, products[:, 4].astype(float) * rng.normal(1.0, 0.18, n)).round(2)
    product_fragility = products[:, 5].astype(float)

    promised_delivery_days = np.select(
        [distance_km < 80, distance_km < 300, distance_km < 900, distance_km < 1800],
        [1, 2, 4, 6],
        default=8,
    ).astype(int)

    carrier_base_risk = carriers[:, 1].astype(float)
    carrier_speed_factor = carriers[:, 2].astype(float)

    estimated_delivery_days = np.ceil(
        (distance_km / 420) / carrier_speed_factor
        + np.where(route_type == "remote", 1.5, 0)
        + np.where(weather == "storm", 1.2, 0)
        + np.where(weather == "rain", 0.35, 0)
        + np.where(traffic == "severe", 0.8, 0)
        + np.where(traffic == "high", 0.35, 0)
        + np.where(is_weekend == 1, 0.35, 0)
    ).astype(int)
    estimated_delivery_days = np.maximum(1, estimated_delivery_days)

    logit = (
        -3.0
        + 0.00115 * distance_km
        + 0.055 * order_weight_kg
        + 1.05 * (weather == "storm").astype(float)
        + 0.42 * (weather == "rain").astype(float)
        + 0.88 * (traffic == "severe").astype(float)
        + 0.34 * (traffic == "high").astype(float)
        + 0.48 * (route_type == "remote").astype(float)
        + 0.23 * (route_type == "long_haul").astype(float)
        + 3.4 * carrier_base_risk
        + 4.0 * product_fragility
        + 0.25 * is_weekend
        + 0.18 * is_peak_hour
        + np.where(estimated_delivery_days > promised_delivery_days, 0.55, 0)
    )
    delay_probability = 1 / (1 + np.exp(-logit))
    delay_risk_label = (rng.random(n) < delay_probability).astype(int)

    delay_days = np.where(
        delay_risk_label == 1,
        rng.integers(1, 4, n) + (weather == "storm").astype(int) + (route_type == "remote").astype(int),
        0,
    )
    actual_delivery_days = estimated_delivery_days + delay_days
    delivery_status = np.where(delay_risk_label == 1, "delayed", "on_time")

    batch_id = np.array([f"HIST-{pd.Timestamp.utcnow().strftime('%Y%m%d')}-{start_idx // max(n,1):06d}"] * n, dtype=object)
    ingestion_ts = pd.Timestamp.utcnow().tz_localize(None)

    streets = rng.choice(["Rua das Flores", "Av. Brasil", "Rua XV de Novembro", "Av. Paulista", "Rua Parana", "Rua Amazonas"], n)
    numbers = rng.integers(10, 9999, n).astype(str)
    address = np.char.add(np.char.add(streets.astype(str), ", "), numbers.astype(str))

    return pd.DataFrame({
        "order_id": [str(uuid.uuid4()) for _ in range(n)],
        "batch_id": batch_id,
        "order_timestamp": pd.to_datetime(order_ts).dt.tz_localize(None),
        "ingestion_timestamp": ingestion_ts,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "city": cities[:, 0],
        "state": cities[:, 1],
        "region": cities[:, 4],
        "address": address,
        "latitude": customer_lat.round(6),
        "longitude": customer_lon.round(6),
        "product_id": products[:, 0],
        "product_bought": products[:, 1],
        "product_category": products[:, 2],
        "order_value": order_value,
        "order_weight_kg": order_weight_kg,
        "product_fragility_score": product_fragility,
        "dc_id": dcs[:, 0],
        "dc_city": dcs[:, 1],
        "dc_state": dcs[:, 2],
        "dc_latitude": dc_lat,
        "dc_longitude": dc_lon,
        "distance_km": distance_km.round(2),
        "carrier": carriers[:, 0],
        "carrier_base_delay_rate": carrier_base_risk,
        "promised_delivery_days": promised_delivery_days,
        "estimated_delivery_days": estimated_delivery_days,
        "actual_delivery_days": actual_delivery_days,
        "delivery_status": delivery_status,
        "delay_risk_label": delay_risk_label,
        "delay_probability_true": delay_probability.round(5),
        "order_year": order_ts.dt.year.astype(int),
        "order_month": order_ts.dt.month.astype(int),
        "order_day": order_ts.dt.day.astype(int),
        "order_day_of_week": dow.astype(int),
        "order_hour": hour.astype(int),
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "weather_condition": weather,
        "traffic_condition": traffic,
        "route_type": route_type,
    })


def init_sqlite(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        batch_id TEXT,
        order_timestamp TEXT,
        ingestion_timestamp TEXT,
        customer_id TEXT,
        customer_name TEXT,
        customer_email TEXT,
        city TEXT,
        state TEXT,
        region TEXT,
        address TEXT,
        latitude REAL,
        longitude REAL,
        product_id TEXT,
        product_bought TEXT,
        product_category TEXT,
        order_value REAL,
        order_weight_kg REAL,
        product_fragility_score REAL,
        dc_id TEXT,
        dc_city TEXT,
        dc_state TEXT,
        dc_latitude REAL,
        dc_longitude REAL,
        distance_km REAL,
        carrier TEXT,
        carrier_base_delay_rate REAL,
        promised_delivery_days INTEGER,
        estimated_delivery_days INTEGER,
        actual_delivery_days INTEGER,
        delivery_status TEXT,
        delay_risk_label INTEGER,
        delay_probability_true REAL,
        order_year INTEGER,
        order_month INTEGER,
        order_day INTEGER,
        order_day_of_week INTEGER,
        order_hour INTEGER,
        is_weekend INTEGER,
        is_peak_hour INTEGER,
        weather_condition TEXT,
        traffic_condition TEXT,
        route_type TEXT
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(order_timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_delay ON orders(delay_risk_label);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_carrier ON orders(carrier);")
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--min-daily", type=int, default=1000)
    parser.add_argument("--max-daily", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--db-path", default="commerceflow.db")
    parser.add_argument("--replace", action="store_true", help="Drop and recreate the orders table before loading.")
    parser.add_argument("--parquet-dir", default=None, help="Optional directory to save parquet chunks.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else None
    if parquet_dir:
        parquet_dir.mkdir(parents=True, exist_ok=True)

    daily = build_daily_counts(args.rows, args.start_date, args.min_daily, args.max_daily, args.seed)

    print("=" * 80)
    print("CommerceFlow AI Historical Generator")
    print(f"Rows: {args.rows:,}")
    print(f"Start date: {args.start_date}")
    print(f"End date: {daily['date'].max().date()}")
    print(f"Daily orders: {args.min_daily:,}-{args.max_daily:,}")
    print(f"SQLite DB: {db_path.resolve()}")
    print(f"Parquet dir: {parquet_dir.resolve() if parquet_dir else 'disabled'}")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    if args.replace:
        print("Dropping existing orders table...")
        conn.execute("DROP TABLE IF EXISTS orders;")
        conn.commit()

    init_sqlite(conn)

    loaded = 0
    chunk_id = 0
    day_pointer = 0

    while loaded < args.rows:
        selected_days = []
        rows_in_chunk = 0
        while day_pointer < len(daily) and rows_in_chunk < args.chunk_size:
            selected_days.append(daily.iloc[day_pointer])
            rows_in_chunk += int(daily.iloc[day_pointer]["orders"])
            day_pointer += 1

        daily_slice = pd.DataFrame(selected_days)
        n = int(daily_slice["orders"].sum())
        if loaded + n > args.rows:
            n = args.rows - loaded
            daily_slice = daily_slice.copy()
            daily_slice.iloc[-1, daily_slice.columns.get_loc("orders")] -= (int(daily_slice["orders"].sum()) - n)

        df = create_orders_chunk(loaded, n, args.start_date, daily_slice, seed=args.seed)

        df_sql = df.copy()
        df_sql["order_timestamp"] = df_sql["order_timestamp"].astype(str)
        df_sql["ingestion_timestamp"] = df_sql["ingestion_timestamp"].astype(str)
        df_sql.to_sql("orders", conn, if_exists="append", index=False)

        if parquet_dir:
            pq = parquet_dir / f"orders_chunk_{chunk_id:05d}.parquet"
            df.to_parquet(pq, index=False)

        loaded += n
        chunk_id += 1
        print(f"Loaded {loaded:,}/{args.rows:,} rows | chunk={chunk_id} | delay_rate={df['delay_risk_label'].mean():.2%}")

    conn.commit()
    conn.close()
    print("DONE.")


if __name__ == "__main__":
    main()
