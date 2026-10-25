"""Part B - delivery-hours regression: split, leakage-free scaling, baseline, four models, CV, one test score."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)


def regression_scores(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["delivery_hours"]

# --- 1. 70 / 15 / 15 split ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- 2. scaler fitted on training rows only ---
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)
pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                        round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
                        round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)], dtype=object),
}).to_csv("outputs/part_b_checks.csv", index=False)

# --- 3. baseline: predict the training mean ---
metrics = [{"model": "mean_baseline", "split": "val",
            **regression_scores(y_val, np.full(len(y_val), y_train.mean()))}]

# --- 4. tree-depth diagnostic ---
depth_rows = []
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))), 4),
        "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))), 4),
        "depth": depth,
    })
depth_table = pd.DataFrame(depth_rows)
depth_table.drop(columns="depth").to_csv("outputs/part_b_tree_depth.csv", index=False)
best_depth = depth_table.loc[depth_table["val_rmse"].idxmin(), "depth"]   # first minimum = smaller depth on ties
best_depth = None if pd.isna(best_depth) else int(best_depth)

# --- 5. four candidate models ---
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    metrics.append({"model": name, "split": "val", **regression_scores(y_val, model.predict(X_val))})

# --- 6. learning-rate experiment ---
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    scores = regression_scores(y_val, gb.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": scores["RMSE"], "val_r2": scores["R2"]})
pd.DataFrame(lr_rows).round(4).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# --- 7. cross-validate the two best, then score the winner on test once ---
val_table = pd.DataFrame(metrics[1:]).sort_values("RMSE", kind="stable")
top_two = val_table["model"].head(2).tolist()

X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in top_two:
    cv_scores = cross_val_score(candidates[name], X_trval, y_trval, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": cv_scores.mean(), "std_r2": cv_scores.std()})
pd.DataFrame(cv_rows).round(4).to_csv("outputs/part_b_cv.csv", index=False)

final_name = top_two[0]
final_model = candidates[final_name]   # already fitted on the training rows
metrics.append({"model": final_name, "split": "test", **regression_scores(y_test, final_model.predict(X_test))})
pd.DataFrame(metrics).round(4).to_csv("outputs/part_b_metrics.csv", index=False)

print(f"split sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"best tree depth: {'unlimited' if best_depth is None else best_depth}")
print(f"final model: {final_name}")
