import json
import random
from pathlib import Path

from eit_engine import compute_eit_components


def simulate_sample():
    features = {
        "hours_awake": random.uniform(0, 24),
        "sleep_hours": random.uniform(3, 9),
        "eye_strain": random.uniform(0, 1),
        "reaction_time": random.uniform(220, 900),
        "hydration": random.uniform(0.2, 1.0),
        "heart_rate": random.uniform(55, 120),
        "steering_variability": random.uniform(0, 1),
        "lane_drift": random.uniform(0, 1),
        "time_of_day": random.uniform(0, 24),
        "food_intake": random.uniform(0, 12),
    }

    gt = (
        0.22 * (features["hours_awake"] / 24.0)
        + 0.18 * (1.0 - min(features["sleep_hours"] / 8.0, 1.0))
        + 0.14 * features["eye_strain"]
        + 0.16 * min((features["reaction_time"] - 200.0) / 800.0, 1.0)
        + 0.08 * (1.0 - features["hydration"])
        + 0.10 * features["steering_variability"]
        + 0.10 * features["lane_drift"]
        + 0.02 * (1.0 if features["time_of_day"] < 6 or features["time_of_day"] > 22 else 0.4)
    )
    return features, max(0.0, min(1.0, gt))


def predict_risk_from_components(theta, lambd, f_state, load, alpha, beta):
    sigma = theta * lambd * f_state
    raw_risk = (load ** alpha) / ((sigma + 1e-6) ** beta)
    return raw_risk / (1.0 + raw_risk)


def loss_for(alpha, beta, dataset):
    err = 0.0
    for features, y_true in dataset:
        theta, lambd, f_state, load = compute_eit_components(features)
        y_pred = predict_risk_from_components(theta, lambd, f_state, load, alpha, beta)
        err += (y_pred - y_true) ** 2
    return err / len(dataset)


def grid_search(dataset):
    best_alpha, best_beta, best_loss = 1.2, 1.5, float("inf")
    for alpha in [0.6 + i * 0.1 for i in range(19)]:
        for beta in [0.6 + i * 0.1 for i in range(19)]:
            current = loss_for(alpha, beta, dataset)
            if current < best_loss:
                best_alpha, best_beta, best_loss = alpha, beta, current
    return best_alpha, best_beta, best_loss


def main():
    dataset = [simulate_sample() for _ in range(2000)]
    alpha, beta, mse = grid_search(dataset)
    out = {"alpha": alpha, "beta": beta, "mse": mse, "samples": len(dataset)}
    Path("model_params.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved model_params.json")
    print(out)


if __name__ == "__main__":
    main()
