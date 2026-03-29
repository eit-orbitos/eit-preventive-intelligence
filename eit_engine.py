from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple


FEATURES = {
    "hours_awake": {"min": 0.0, "max": 24.0},
    "sleep_hours": {"min": 0.0, "max": 12.0},
    "eye_strain": {"min": 0.0, "max": 1.0},
    "reaction_time": {"min": 200.0, "max": 1000.0},
    "hydration": {"min": 0.0, "max": 1.0},
    "heart_rate": {"min": 50.0, "max": 140.0},
    "steering_variability": {"min": 0.0, "max": 1.0},
    "lane_drift": {"min": 0.0, "max": 1.0},
    "time_of_day": {"min": 0.0, "max": 24.0},
    "food_intake": {"min": 0.0, "max": 12.0},
}

EIT_MAPPING = {
    "Theta": ["steering_variability", "lane_drift"],
    "Lambda": ["reaction_time"],
    "F": [
        "hours_awake",
        "sleep_hours",
        "eye_strain",
        "hydration",
        "heart_rate",
        "food_intake",
    ],
    "Load": ["time_of_day"],
}

FEATURE_DIRECTIONS = {
    "hours_awake": "higher_worse",
    "sleep_hours": "lower_worse",
    "eye_strain": "higher_worse",
    "reaction_time": "higher_worse",
    "hydration": "lower_worse",
    "heart_rate": "extreme_worse",
    "steering_variability": "higher_worse",
    "lane_drift": "higher_worse",
    "time_of_day": "night_worse",
    "food_intake": "higher_worse",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_raw(name: str, value: float) -> float:
    spec = FEATURES[name]
    min_v = spec["min"]
    max_v = spec["max"]
    if max_v <= min_v:
        return 0.0
    return clamp((value - min_v) / (max_v - min_v))


def score_feature(name: str, value: float) -> float:
    direction = FEATURE_DIRECTIONS[name]
    x = normalize_raw(name, value)

    if direction == "higher_worse":
        return 1.0 - x
    if direction == "lower_worse":
        return x
    if direction == "extreme_worse":
        return max(0.0, 1.0 - abs(x - 0.5) / 0.5)
    if direction == "night_worse":
        h = value % 24.0
        if 6 <= h < 18:
            return 1.0
        if 18 <= h < 22:
            return 0.7
        if 4 <= h < 6 or 22 <= h < 24:
            return 0.4
        return 0.2
    return 0.5


def average_scores(features: Dict[str, float], names: List[str]) -> float:
    vals = [score_feature(name, features[name]) for name in names]
    return sum(vals) / len(vals) if vals else 0.0


def compute_eit_components(features: Dict[str, float]) -> Tuple[float, float, float, float]:
    theta = average_scores(features, EIT_MAPPING["Theta"])
    lambd = average_scores(features, EIT_MAPPING["Lambda"])
    f_state = average_scores(features, EIT_MAPPING["F"])
    load_resilience = average_scores(features, EIT_MAPPING["Load"])
    load = 1.0 - load_resilience
    return clamp(theta), clamp(lambd), clamp(f_state), clamp(load)


def classify_status(risk: float) -> str:
    if risk < 0.30:
        return "GREEN"
    if risk < 0.60:
        return "YELLOW"
    if risk < 0.80:
        return "ORANGE"
    return "RED"


def recommended_action(status: str) -> str:
    return {
        "GREEN": "Continue",
        "YELLOW": "Monitor closely",
        "ORANGE": "Prepare to stop and rest",
        "RED": "Stop immediately",
    }[status]


def compute_risk(features, alpha=1.2, beta=1.5, epsilon=1e-6):
    theta, lambd, f_state, load = compute_eit_components(features)
    sigma = theta * lambd * f_state
    raw_risk = (load ** alpha) / ((sigma + epsilon) ** beta)
    risk = raw_risk / (1.0 + raw_risk)
    status = classify_status(risk)

    factors = {
        "Theta": 1.0 - theta,
        "Lambda": 1.0 - lambd,
        "F": 1.0 - f_state,
        "Load": load,
    }
    dominant_factor = max(factors, key=factors.get)

    return {
        "Theta": round(theta, 4),
        "Lambda": round(lambd, 4),
        "F": round(f_state, 4),
        "Load": round(load, 4),
        "Sigma": round(sigma, 4),
        "RawRisk": round(raw_risk, 4),
        "Risk": round(risk, 4),
        "Status": status,
        "Confidence": round(sigma, 4),
        "DominantFactor": dominant_factor,
        "RecommendedAction": recommended_action(status),
        "Alpha": round(alpha, 4),
        "Beta": round(beta, 4),
    }


class RiskTrendBuffer:
    def __init__(self, maxlen: int = 60):
        self.values: Deque[float] = deque(maxlen=maxlen)

    def update(self, risk: float):
        self.values.append(risk)
        vals = list(self.values)
        current = vals[-1]
        prev = vals[-2] if len(vals) >= 2 else current
        delta = current - prev

        short_avg = sum(vals[-5:]) / min(len(vals), 5)
        long_avg = sum(vals[-15:]) / min(len(vals), 15)
        slope = short_avg - long_avg

        if slope > 0.08 or delta > 0.12:
            trend = "RISING_FAST"
        elif slope > 0.02 or delta > 0.04:
            trend = "RISING"
        elif slope < -0.08 or delta < -0.12:
            trend = "FALLING_FAST"
        elif slope < -0.02 or delta < -0.04:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "CurrentRisk": round(current, 4),
            "Delta": round(delta, 4),
            "ShortAvg": round(short_avg, 4),
            "LongAvg": round(long_avg, 4),
            "SlopeSignal": round(slope, 4),
            "Trend": trend,
            "Series": [round(v, 4) for v in vals],
        }


def build_alerts(prediction: dict, trend: dict, camera: dict) -> dict:
    status = prediction["Status"]
    trend_name = trend["Trend"]
    alerts = []

    if status == "YELLOW":
        alerts.append("Fatigue risk rising. Increase caution.")
    elif status == "ORANGE":
        alerts.append("High fatigue risk. Prepare to stop and rest.")
    elif status == "RED":
        alerts.append("Critical fatigue risk. Stop immediately.")

    cv = camera.get("cv_features") if camera else None
    if cv and cv.get("drowsiness", 0) >= 0.75:
        alerts.append("Eyes indicate strong drowsiness.")
    if cv and cv.get("eyes_closed_frames", 0) >= 8:
        alerts.append("Possible microsleep detected.")
    if trend_name in {"RISING", "RISING_FAST"}:
        alerts.append(f"Risk trend is {trend_name.lower().replace('_', ' ')}.")

    return {
        "messages": alerts,
        "sound": status in {"ORANGE", "RED"},
        "vibration": status == "RED",
    }
