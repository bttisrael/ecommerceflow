from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from faker import Faker

fake = Faker("pt_BR")

BRAZIL_CITIES = [
    {"city": "Curitiba", "state": "PR", "region": "South", "lat": -25.4284, "lon": -49.2733},
    {"city": "Sao Paulo", "state": "SP", "region": "Southeast", "lat": -23.5505, "lon": -46.6333},
    {"city": "Rio de Janeiro", "state": "RJ", "region": "Southeast", "lat": -22.9068, "lon": -43.1729},
    {"city": "Belo Horizonte", "state": "MG", "region": "Southeast", "lat": -19.9167, "lon": -43.9345},
    {"city": "Porto Alegre", "state": "RS", "region": "South", "lat": -30.0346, "lon": -51.2177},
    {"city": "Florianopolis", "state": "SC", "region": "South", "lat": -27.5949, "lon": -48.5482},
    {"city": "Goiania", "state": "GO", "region": "Central-West", "lat": -16.6869, "lon": -49.2648},
    {"city": "Brasilia", "state": "DF", "region": "Central-West", "lat": -15.7939, "lon": -47.8828},
    {"city": "Salvador", "state": "BA", "region": "Northeast", "lat": -12.9777, "lon": -38.5016},
    {"city": "Recife", "state": "PE", "region": "Northeast", "lat": -8.0476, "lon": -34.8770},
    {"city": "Fortaleza", "state": "CE", "region": "Northeast", "lat": -3.7319, "lon": -38.5267},
    {"city": "Manaus", "state": "AM", "region": "North", "lat": -3.1190, "lon": -60.0217},
    {"city": "Belem", "state": "PA", "region": "North", "lat": -1.4558, "lon": -48.4902},
]

DISTRIBUTION_CENTERS = [
    {"dc_id": "DC_SP", "dc_city": "Sao Paulo", "dc_state": "SP", "dc_latitude": -23.5505, "dc_longitude": -46.6333},
    {"dc_id": "DC_PR", "dc_city": "Curitiba", "dc_state": "PR", "dc_latitude": -25.4284, "dc_longitude": -49.2733},
    {"dc_id": "DC_PE", "dc_city": "Recife", "dc_state": "PE", "dc_latitude": -8.0476, "dc_longitude": -34.8770},
    {"dc_id": "DC_GO", "dc_city": "Goiania", "dc_state": "GO", "dc_latitude": -16.6869, "dc_longitude": -49.2648},
]

PRODUCTS = [
    {"product_id": "P001", "product_bought": "Smartphone Orion X", "product_category": "Electronics", "base_value": 2499.90, "weight": 0.45, "fragility": 0.06},
    {"product_id": "P002", "product_bought": "Wireless Headphones", "product_category": "Electronics", "base_value": 399.90, "weight": 0.25, "fragility": 0.04},
    {"product_id": "P003", "product_bought": "Premium Coffee Pack", "product_category": "Grocery", "base_value": 89.90, "weight": 1.20, "fragility": 0.02},
    {"product_id": "P004", "product_bought": "Skin Care Kit", "product_category": "Beauty", "base_value": 279.90, "weight": 0.65, "fragility": 0.03},
    {"product_id": "P005", "product_bought": "Running Shoes", "product_category": "Fashion", "base_value": 349.90, "weight": 0.90, "fragility": 0.03},
    {"product_id": "P006", "product_bought": "Robot Vacuum", "product_category": "Home", "base_value": 1599.90, "weight": 4.80, "fragility": 0.08},
    {"product_id": "P007", "product_bought": "Office Chair", "product_category": "Home", "base_value": 899.90, "weight": 12.50, "fragility": 0.12},
    {"product_id": "P008", "product_bought": "Baby Care Bundle", "product_category": "Baby", "base_value": 219.90, "weight": 2.20, "fragility": 0.04},
    {"product_id": "P009", "product_bought": "Gaming Keyboard", "product_category": "Electronics", "base_value": 599.90, "weight": 0.85, "fragility": 0.05},
    {"product_id": "P010", "product_bought": "Protein Pack", "product_category": "Health", "base_value": 149.90, "weight": 1.80, "fragility": 0.03},
    {"product_id": "P011", "product_bought": "Air Fryer", "product_category": "Home", "base_value": 499.90, "weight": 5.10, "fragility": 0.07},
    {"product_id": "P012", "product_bought": "Smart Watch", "product_category": "Electronics", "base_value": 899.90, "weight": 0.30, "fragility": 0.05},
]

CARRIERS = [
    {"carrier": "BlueExpress", "carrier_base_delay_rate": 0.055, "speed_factor": 1.00},
    {"carrier": "RoadRunner", "carrier_base_delay_rate": 0.075, "speed_factor": 0.95},
    {"carrier": "EcoFrete", "carrier_base_delay_rate": 0.105, "speed_factor": 0.85},
    {"carrier": "FlashLog", "carrier_base_delay_rate": 0.090, "speed_factor": 1.10},
    {"carrier": "NationalPost", "carrier_base_delay_rate": 0.130, "speed_factor": 0.75},
]

WEATHER = ["clear", "rain", "storm", "fog", "heatwave"]
ROUTE_TYPES = ["urban", "regional", "long_haul", "remote"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_dc(customer_lat: float, customer_lon: float) -> Dict[str, Any]:
    ranked = []
    for dc in DISTRIBUTION_CENTERS:
        d = haversine_km(customer_lat, customer_lon, dc["dc_latitude"], dc["dc_longitude"])
        ranked.append((d, dc))
    ranked.sort(key=lambda x: x[0])

    # Simulate capacity overflow: 15% of orders are shipped from a non-nearest DC.
    if random.random() < 0.15:
        _, dc = random.choice(ranked)
        dist = haversine_km(customer_lat, customer_lon, dc["dc_latitude"], dc["dc_longitude"])
    else:
        dist, dc = ranked[0]

    out = dict(dc)
    out["distance_km"] = round(dist, 2)
    return out


def jitter_coordinate(value: float, scale: float = 0.13) -> float:
    return value + random.uniform(-scale, scale)


def choose_traffic(order_hour: int) -> str:
    if order_hour in [8, 9, 10, 11, 12, 18, 19, 20, 21]:
        return random.choices(["low", "medium", "high", "severe"], weights=[12, 38, 36, 14])[0]
    return random.choices(["low", "medium", "high", "severe"], weights=[35, 43, 17, 5])[0]


def calculate_delivery_features(
    *,
    order_dt: datetime,
    distance_km: float,
    weight_kg: float,
    product_fragility_score: float,
    carrier: Dict[str, Any],
    weather_condition: str,
    traffic_condition: str,
    route_type: str,
) -> Dict[str, Any]:
    is_weekend = 1 if order_dt.weekday() >= 5 else 0
    is_peak_hour = 1 if order_dt.hour in [8, 9, 10, 11, 12, 18, 19, 20, 21] else 0

    promised_delivery_days = 1 if distance_km < 80 else 2 if distance_km < 300 else 4 if distance_km < 900 else 6 if distance_km < 1800 else 8

    weather_delay = 1.2 if weather_condition == "storm" else 0.35 if weather_condition == "rain" else 0.0
    traffic_delay = 0.8 if traffic_condition == "severe" else 0.35 if traffic_condition == "high" else 0.0
    route_delay = 1.5 if route_type == "remote" else 0.0

    estimated_delivery_days = math.ceil(
        (distance_km / 420.0) / carrier["speed_factor"]
        + route_delay
        + weather_delay
        + traffic_delay
        + (0.35 if is_weekend else 0.0)
    )
    estimated_delivery_days = max(1, estimated_delivery_days)

    logit = (
        -3.0
        + 0.00115 * distance_km
        + 0.055 * weight_kg
        + 1.05 * (weather_condition == "storm")
        + 0.42 * (weather_condition == "rain")
        + 0.88 * (traffic_condition == "severe")
        + 0.34 * (traffic_condition == "high")
        + 0.48 * (route_type == "remote")
        + 0.23 * (route_type == "long_haul")
        + 3.4 * carrier["carrier_base_delay_rate"]
        + 4.0 * product_fragility_score
        + 0.25 * is_weekend
        + 0.18 * is_peak_hour
        + (0.55 if estimated_delivery_days > promised_delivery_days else 0.0)
    )

    delay_probability_true = 1.0 / (1.0 + math.exp(-logit))
    delay_risk_label = 1 if random.random() < delay_probability_true else 0
    delay_days = random.randint(1, 3) + (1 if weather_condition == "storm" else 0) + (1 if route_type == "remote" else 0) if delay_risk_label else 0
    actual_delivery_days = estimated_delivery_days + delay_days

    return {
        "promised_delivery_days": promised_delivery_days,
        "estimated_delivery_days": estimated_delivery_days,
        "actual_delivery_days": actual_delivery_days,
        "delivery_status": "delayed" if delay_risk_label else "on_time",
        "delay_risk_label": delay_risk_label,
        "delay_probability_true": round(delay_probability_true, 5),
        "order_year": order_dt.year,
        "order_month": order_dt.month,
        "order_day": order_dt.day,
        "order_day_of_week": order_dt.weekday(),
        "order_hour": order_dt.hour,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
    }


def generate_order(batch_id: str | None = None, order_dt: datetime | None = None) -> Dict[str, Any]:
    city = random.choice(BRAZIL_CITIES)
    product = random.choice(PRODUCTS)
    carrier = random.choice(CARRIERS)
    order_dt = order_dt or datetime.now(timezone.utc)

    lat = jitter_coordinate(city["lat"])
    lon = jitter_coordinate(city["lon"])
    dc = nearest_dc(lat, lon)

    weather_condition = random.choices(WEATHER, weights=[62, 24, 6, 4, 4])[0]
    traffic_condition = choose_traffic(order_dt.hour)
    route_type = random.choices(ROUTE_TYPES, weights=[38, 32, 22, 8])[0]

    order_weight_kg = max(0.05, random.gauss(product["weight"], product["weight"] * 0.18))
    order_value = max(19.9, random.gauss(product["base_value"], product["base_value"] * 0.12))
    if order_dt.month in [11, 12]:
        order_value *= 1.08

    delivery = calculate_delivery_features(
        order_dt=order_dt,
        distance_km=dc["distance_km"],
        weight_kg=order_weight_kg,
        product_fragility_score=product["fragility"],
        carrier=carrier,
        weather_condition=weather_condition,
        traffic_condition=traffic_condition,
        route_type=route_type,
    )

    customer_name = fake.name()
    first_email = customer_name.lower().replace(" ", ".").replace("'", "").replace(",", "")
    email = f"{first_email}.{random.randint(100, 9999)}@example.com"

    return {
        "order_id": str(uuid.uuid4()),
        "batch_id": batch_id or str(uuid.uuid4()),
        "order_timestamp": order_dt.isoformat(),
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),

        "customer_id": f"CUST-LIVE-{random.randint(1, 999999999):09d}",
        "customer_name": customer_name,
        "customer_email": email,
        "city": city["city"],
        "state": city["state"],
        "region": city["region"],
        "address": fake.street_address(),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),

        "product_id": product["product_id"],
        "product_bought": product["product_bought"],
        "product_category": product["product_category"],
        "order_value": round(order_value, 2),
        "order_weight_kg": round(order_weight_kg, 2),
        "product_fragility_score": product["fragility"],

        **dc,

        "carrier": carrier["carrier"],
        "carrier_base_delay_rate": carrier["carrier_base_delay_rate"],
        **delivery,

        "weather_condition": weather_condition,
        "traffic_condition": traffic_condition,
        "route_type": route_type,
    }


def generate_orders(n: int = 3000, batch_id: str | None = None) -> List[Dict[str, Any]]:
    batch_id = batch_id or str(uuid.uuid4())
    return [generate_order(batch_id=batch_id) for _ in range(n)]
