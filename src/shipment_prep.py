"""Shared constants and the cleaning / feature functions used by every Part and by predict_delay.py."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
MAX_VALID_WEIGHT_KG = 100   # anything above this is the scanner-fault sentinel (999.0)


def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians).

    If medians is None the fill-in medians are computed from this table,
    otherwise the given dictionary is used (for new parcels at scoring time).
    """
    # 1. exact duplicate rows
    df = raw.drop_duplicates().reset_index(drop=True)

    # 2. one spelling per service type
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. blank payment mode -> placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. "₹1,290" -> 1290.0
    text = df["declared_value"].str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(text)
    df = df.drop(columns=["declared_value"])

    # 5. dispatch time as datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")

    # 6. weight sentinel -> missing
    df.loc[df["parcel_weight_kg"] > MAX_VALID_WEIGHT_KG, "parcel_weight_kg"] = np.nan

    # 7. median fill
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub attributes and derive the model features. Does not use delivery_hours."""
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    df = clean.merge(hubs[hub_cols], on="hub_id", how="left", validate="many_to_one")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
