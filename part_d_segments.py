"""Part D (segments) - hub profile, K-Means operating segments, separation evidence and the tier check."""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENT_NAMES = ["Standard", "Watch", "Escalate"]          # lowest -> highest mean delay rate
TIER_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def across_over_within(Z, labels):
    """Mean distance of hub pairs in different clusters / mean distance of pairs in the same cluster."""
    i, j = np.triu_indices(len(Z), k=1)
    dist = np.linalg.norm(Z[i] - Z[j], axis=1)
    same = labels[i] == labels[j]
    return dist[~same].mean() / dist[same].mean()


table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# --- 1. hub profile, ascending hub_id ---
profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "count"), mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"), mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index()
           .sort_values("hub_id")
           .reset_index(drop=True))
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

# --- 2. scale and cluster (K = 3 is fixed by the business) ---
Z = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])

# --- 3. separation evidence for K = 2..5 ---
separation_rows = []
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    sizes = sorted(np.bincount(labels), reverse=True)
    separation_rows.append({"K": k, "across_over_within": round(float(across_over_within(Z, labels)), 4),
                            "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

# --- 4. name the K = 3 clusters by mean delay rate and check them against the tier label ---
profile["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Z)
cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index
profile["segment"] = profile["cluster"].map(dict(zip(cluster_order, SEGMENT_NAMES)))

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
segments.to_csv("outputs/hub_segments.csv", index=False)

segment_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "count"), **{c: (c, "mean") for c in PROFILE_COLUMNS})
                   .reindex(SEGMENT_NAMES)
                   .reset_index())
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

mismatch = segments[segments["segment"] != segments["hub_tier"].map(TIER_SEGMENT)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(segment_profile.round(2).to_string(index=False))
print("tier label does not match behaviour for:", ", ".join(mismatch["hub_id"] + " " + mismatch["hub_city"]))
