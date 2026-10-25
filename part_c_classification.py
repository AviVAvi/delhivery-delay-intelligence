"""Part C - delay classifier: baselines, feature signal check, forest tuning, alert threshold, one test score."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SELECTOR_K = 7
MIN_RECALL = 0.80


def classification_scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]
y = table["is_delayed"]

# --- 1. stratified 70 / 15 / 15 split ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"split sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}, "
      f"delayed: {y_train.sum()} / {y_val.sum()} / {y_test.sum()}")

# --- 2. blind baseline and three classifiers at the default 0.5 rule ---
model_rows = [{"model": "blind_all_on_time", **classification_scores(y_val, np.zeros(len(y_val), dtype=int))}]

logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **classification_scores(y_val, logreg.predict(X_val))})

tree_f1 = {}
for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42).fit(X_train, y_train)
    tree_f1[depth] = round(f1_score(y_val, tree.predict(X_val)), 4)
best_tree_depth = max(tree_f1, key=lambda d: (tree_f1[d], -d))   # highest F1, smaller depth on ties
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_tree_depth, random_state=42).fit(X_train, y_train)
model_rows.append({"model": f"decision_tree_depth_{best_tree_depth}", **classification_scores(y_val, tree.predict(X_val))})

forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **classification_scores(y_val, forest.predict(X_val))})
pd.DataFrame(model_rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# --- 3. which columns carry signal? ---
# `forest` above is exactly the default forest on all 17 training columns
f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
feature_scores = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                               "rf_importance": forest.feature_importances_})
feature_scores["f_rank"] = feature_scores["f_score"].rank(ascending=False, method="first").astype(int)
feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)
feature_scores = feature_scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
feature_scores.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
pd.DataFrame(selectk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# --- 4. tuning on the 7 selected columns (selector fitted on training rows only) ---
selector = SelectKBest(f_classif, k=SELECTOR_K).fit(X_train, y_train)
X_train_k = selector.transform(X_train)
X_val_k = selector.transform(X_val)

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_train_k, y_train)
randomized = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                                n_iter=9, cv=5, random_state=42).fit(X_train_k, y_train)
default_k = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_k, y_train)

default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
tuning = pd.DataFrame([
    {"search": "grid_search", "candidates": len(grid.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(grid.cv_results_["params"]) * 5, "best_params": str(grid.best_params_),
     "best_cv_accuracy": grid.best_score_, "val_f1": f1_score(y_val, grid.predict(X_val_k))},
    {"search": "randomized_search", "candidates": len(randomized.cv_results_["params"]), "cv_folds": 5,
     "total_fits": len(randomized.cv_results_["params"]) * 5, "best_params": str(randomized.best_params_),
     "best_cv_accuracy": randomized.best_score_, "val_f1": f1_score(y_val, randomized.predict(X_val_k))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "defaults", "best_cv_accuracy": np.nan, "val_f1": f1_score(y_val, default_k.predict(X_val_k))},
]).round(4)
tuning.to_csv("outputs/part_c_tuning.csv", index=False)

# --- 5. final configuration: highest val_f1, ties -> default, grid, randomized ---
tie_order = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
chosen_from = min(tuning["search"], key=lambda s: (-tuning.set_index("search").loc[s, "val_f1"], tie_order[s]))
chosen_params = {"grid_search": grid.best_params_, "randomized_search": randomized.best_params_,
                 "default_forest_same_7_features": default_params}[chosen_from]

final_model = Pipeline([("selector", SelectKBest(f_classif, k=SELECTOR_K)),
                        ("model", RandomForestClassifier(random_state=42, **chosen_params))]).fit(X_train, y_train)

# --- 6. alert threshold from validation probabilities ---
val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = [{"threshold": t, **classification_scores(y_val, (val_proba >= t).astype(int))} for t in THRESHOLDS]
thresholds = pd.DataFrame(threshold_rows)
thresholds.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)

meets_recall = thresholds.loc[thresholds["recall"] >= MIN_RECALL, "threshold"]
alert_threshold = float(meets_recall.max()) if len(meets_recall) else 0.1

# --- 7. one test evaluation at the chosen threshold ---
test_proba = final_model.predict_proba(X_test)[:, 1]
final_test = {"threshold": alert_threshold, **classification_scores(y_test, (test_proba >= alert_threshold).astype(int))}
pd.DataFrame([final_test]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)

pd.DataFrame([{
    "selector_k": SELECTOR_K,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": alert_threshold,
    "chosen_from": chosen_from,
}]).to_csv("outputs/part_c_final_config.csv", index=False)

print(f"best decision tree depth: {best_tree_depth}")
print(f"final configuration from: {chosen_from} {chosen_params}")
print(f"alert threshold: {alert_threshold}")
