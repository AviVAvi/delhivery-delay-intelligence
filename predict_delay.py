"""Score a batch of new parcels and turn the delay probabilities into an action list."""
import os

import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

os.makedirs("outputs", exist_ok=True)

new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

clean, _ = clean_shipments(new, medians=medians)
features = build_features(clean, hubs)

scored = features[["shipment_id", "hub_id"]].copy()
scored["delay_probability"] = model.predict_proba(features[feature_order])[:, 1].round(4)
scored["flagged"] = (scored["delay_probability"] >= threshold).astype(int)
scored = scored.merge(segments, on="hub_id", how="left", validate="many_to_one")


def choose_action(row):
    if not row["flagged"]:
        return "No action"
    if row["segment"] == "Escalate":
        return "Escalate to hub manager"
    return "Proactive customer call"


scored["action"] = scored.apply(choose_action, axis=1)
scored = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
scored.to_csv("outputs/new_shipment_actions.csv", index=False)

print(f"{len(scored)} parcels scored, {scored['flagged'].sum()} flagged (threshold {threshold})")
print(scored["action"].value_counts().to_string())
