
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\DS\uknd")

TRAINING_CSV = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "BG_A"
    / "combined_training.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = (
    OUTPUT_DIR / "local_xgboost_spatial_test_predictions.csv"
)

FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "profile_curvature",
    "plan_curvature",
    "land_cover",
    "distance_to_road",
    "distance_to_river",
]

NUMERIC_FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "profile_curvature",
    "plan_curvature",
    "distance_to_road",
    "distance_to_river",
]

BLOCK_SIZE_M = 10_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# This is the same threshold used in the earlier evaluation example.
CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("LOCAL XGBOOST SPATIAL TEST EVALUATION")
print("=" * 65)

if not TRAINING_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found: {TRAINING_CSV}"
    )

df = pd.read_csv(TRAINING_CSV)

required = set(FEATURES) | {
    "target",
    "x_utm44n",
    "y_utm44n",
}
missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Training CSV is missing columns: {sorted(missing)}"
    )

# Match the filtering logic from predict_susceptibility.py.
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")

y = pd.to_numeric(
    df["target"],
    errors="coerce",
)

coords = df[
    ["x_utm44n", "y_utm44n"]
].apply(pd.to_numeric, errors="coerce")

valid = (
    X.notna().all(axis=1)
    & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    & y.isin([0, 1])
    & coords.notna().all(axis=1)
    & np.isfinite(coords.to_numpy(dtype=float)).all(axis=1)
)

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].astype(int).reset_index(drop=True)
coords = coords.loc[valid].reset_index(drop=True)

print(f"Original rows: {len(df):,}")
print(f"Valid samples: {len(X):,}")
print(f"Removed samples: {len(df) - len(X):,}")
print(f"Class counts: {y.value_counts().sort_index().to_dict()}")

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Both target classes (0 and 1) must be present."
    )


# ============================================================
# 3. RECREATE THE SAME SPATIAL SPLIT
# ============================================================

block_x = np.floor(
    coords["x_utm44n"] / BLOCK_SIZE_M
).astype(int)

block_y = np.floor(
    coords["y_utm44n"] / BLOCK_SIZE_M
).astype(int)

groups = (
    block_x.astype(str)
    + "_"
    + block_y.astype(str)
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

overlap = train_groups & test_groups

if overlap:
    raise RuntimeError(
        f"Spatial group leakage detected: {len(overlap)} groups overlap."
    )

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

print(f"\nTraining samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")
print(f"Training groups: {len(train_groups):,}")
print(f"Test groups: {len(test_groups):,}")
print(f"Overlapping groups: {len(overlap)}")

print(
    "Training class counts:",
    y_train.value_counts().sort_index().to_dict(),
)
print(
    "Test class counts:",
    y_test.value_counts().sort_index().to_dict(),
)

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError(
        "Both classes must occur in the training and test partitions."
    )


# ============================================================
# 4. BUILD THE SAME XGBOOST PIPELINE
# ============================================================

preprocessor = ColumnTransformer([
    (
        "land_cover",
        OneHotEncoder(handle_unknown="ignore"),
        ["land_cover"],
    ),
    (
        "numeric",
        "passthrough",
        NUMERIC_FEATURES,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )),
])


# ============================================================
# 5. FIT ONLY ON THE TRAINING PARTITION
# ============================================================

print("\nFitting XGBoost on training samples only...")
model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# 6. GENERATE SCORES AND TEST METRICS
# ============================================================

train_scores = model.predict_proba(X_train)[:, 1]
test_scores = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_scores >= CLASSIFICATION_THRESHOLD
).astype(int)

# Training metrics are diagnostic and are not the main result.
train_roc_auc = roc_auc_score(y_train, train_scores)
train_ap = average_precision_score(y_train, train_scores)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)

test_roc_auc = roc_auc_score(
    y_test, test_scores
)

test_ap = average_precision_score(
    y_test, test_scores
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TRAINING METRICS — DIAGNOSTIC ONLY")
print("=" * 65)

print(f"Training ROC-AUC:           {train_roc_auc:.4f}")
print(f"Training average precision: {train_ap:.4f}")

print("\n" + "=" * 65)
print("SPATIALLY HELD-OUT TEST METRICS")
print("=" * 65)

print(f"Accuracy:          {test_accuracy:.4f}")
print(f"Balanced accuracy: {test_balanced_accuracy:.4f}")
print(f"ROC-AUC:           {test_roc_auc:.4f}")
print(f"Average precision: {test_ap:.4f}")

print(
    f"\nClassification threshold: "
    f"{CLASSIFICATION_THRESHOLD:.2f}"
)

print("\nConfusion matrix")
print("Rows = true class; columns = predicted class")
print("Labels: 0 = background, 1 = recorded landslide")
print(cm)

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=[
        "Background (unrecorded)",
        "Recorded landslide",
    ],
    digits=4,
    zero_division=0,
))


# ============================================================
# 8. SAVE TEST PREDICTIONS
# ============================================================

test_coords = coords.iloc[test_idx].reset_index(drop=True)

test_groups = groups.iloc[test_idx].reset_index(drop=True)

results = pd.DataFrame({
    "x_utm44n": test_coords["x_utm44n"],
    "y_utm44n": test_coords["y_utm44n"],
    "spatial_group": test_groups,
    "target": y_test.to_numpy(),
    "predicted_score": test_scores,
    "predicted_class_at_threshold": test_predictions,
})

results.to_csv(PREDICTIONS_CSV, index=False)

print(f"\nTest predictions saved to:\n{PREDICTIONS_CSV}")

print("\nEvaluation complete.")
print(
    "Interpret the results as discrimination between recorded "
    "landslides and unrecorded background samples."
)
