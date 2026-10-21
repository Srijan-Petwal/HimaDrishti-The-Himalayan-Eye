
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\DS\uknd")
RASTER_DIR = Path(r"E:\DS\uknd_raster_backup")

TRAINING_CSV = (
    PROJECT_DIR / "data" / "processed" / "BG_A"
    / "combined_training.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "susceptibility"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RASTER = OUTPUT_DIR / "uknd_xgboost_susceptibility.tif"

REFERENCE_RASTER = (
    RASTER_DIR / "dem" / "tiles"
    / "uk_dem_glo30_utm44n_clipped.tif"
)

# Predictor names must match the training CSV.
RASTER_FILES = {
    "elevation": (
        RASTER_DIR / "dem" / "tiles"
        / "uk_dem_glo30_utm44n_clipped.tif"
    ),
    "slope": (
        RASTER_DIR / "dem" / "tiles"
        / "uk_slope_cleaned.tif"
    ),
    "aspect": (
        RASTER_DIR / "ml_aligned"
        / "uk_aspect_dem_grid.tif"
    ),
    "profile_curvature": (
        RASTER_DIR / "terrain"
        / "uk_profile_curvature.tif"
    ),
    "plan_curvature": (
        RASTER_DIR / "terrain"
        / "uk_plan_curvature.tif"
    ),
    "land_cover": (
        RASTER_DIR / "ml_aligned"
        / "uk_worldcover_dem_grid.tif"
    ),
    "distance_to_road": (
        RASTER_DIR / "ROADS"
        / "uk_distance_to_road_final.tif"
    ),
    "distance_to_river": (
        RASTER_DIR / "RIVERS"
        / "uk_distance_to_river_aligned_v2.tif"
    ),
}

FEATURES = list(RASTER_FILES)

BLOCK_SIZE_M = 10_000
TEST_SIZE = 0.20
RANDOM_STATE = 42

WINDOW_SIZE = 512
OUTPUT_NODATA = -9999.0



# ============================================================
# 2. LOAD DATA AND RECREATE THE VALIDATED SPATIAL SPLIT
# ============================================================

print("=" * 65)
print("UTTARAKHAND LANDSLIDE SUSCEPTIBILITY")
print("=" * 65)

if not TRAINING_CSV.exists():
    raise FileNotFoundError(f"Training CSV not found: {TRAINING_CSV}")

df = pd.read_csv(TRAINING_CSV)

required = set(FEATURES) | {
    "target", "x_utm44n", "y_utm44n"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Training CSV is missing: {sorted(missing)}")

# Validate predictors, targets and coordinates.
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["target"], errors="coerce")
coords = df[["x_utm44n", "y_utm44n"]].apply(
    pd.to_numeric, errors="coerce"
)

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

# IMPORTANT: match validate_split.py exactly.
block_size = 10_000

block_x = np.floor(
    coords["x_utm44n"] / block_size
).astype(int)

block_y = np.floor(
    coords["y_utm44n"] / block_size
).astype(int)

groups = block_x.astype(str) + "_" + block_y.astype(str)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert not (train_groups & test_groups), "Spatial group leakage detected!"

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

print(f"Valid samples: {len(X):,}")
print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(test_idx):,}")
print(f"Training groups: {len(train_groups):,}")
print(f"Test groups: {len(test_groups):,}")
print(f"Spatial group overlap: {len(train_groups & test_groups)}")

# Keep the test partition out of model fitting.

# ============================================================
# 3. BUILD AND FIT THE FINAL XGBOOST PIPELINE
# ============================================================

numeric_features = [
    "elevation",
    "slope",
    "aspect",
    "profile_curvature",
    "plan_curvature",
    "distance_to_road",
    "distance_to_river",
]

preprocessor = ColumnTransformer([
    (
        "land_cover",
        OneHotEncoder(handle_unknown="ignore"),
        ["land_cover"],
    ),
    ("numeric", "passthrough", numeric_features),
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

print("\nFitting XGBoost on the training partition...")
model.fit(X_train, y_train)
print("Model fitted.")


# ============================================================
# 4. VALIDATE RASTER GRID
# ============================================================

datasets = {}

try:
    reference = rasterio.open(REFERENCE_RASTER)

    for name, path in RASTER_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} raster: {path}")

        ds = rasterio.open(path)
        datasets[name] = ds

        same_grid = (
            ds.crs == reference.crs
            and ds.width == reference.width
            and ds.height == reference.height
            and np.allclose(
                tuple(ds.transform),
                tuple(reference.transform),
                rtol=0,
                atol=1e-7,
            )
        )

        if not same_grid:
            raise ValueError(
                f"{name} does not match the reference grid. "
                "Resolve alignment before predicting."
            )

    print("\nAll eight raster grids validated.")

    # ========================================================
    # 5. WINDOWED PREDICTION
    # ========================================================

    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    height = reference.height
    width = reference.width

    total_windows = (
        (height + WINDOW_SIZE - 1) // WINDOW_SIZE
    ) * (
        (width + WINDOW_SIZE - 1) // WINDOW_SIZE
    )

    processed_windows = 0
    predicted_pixels = 0
    skipped_pixels = 0

    print(f"Raster dimensions: {width:,} x {height:,}")
    print(f"Processing {total_windows:,} windows...")

    with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:

        for row_off in range(0, height, WINDOW_SIZE):
            win_height = min(WINDOW_SIZE, height - row_off)

            for col_off in range(0, width, WINDOW_SIZE):
                win_width = min(WINDOW_SIZE, width - col_off)

                window = Window(
                    col_off, row_off, win_width, win_height
                )

                arrays = {}
                valid = np.ones(
                    (win_height, win_width), dtype=bool
                )

                # Read values and validity masks from all predictors.
                for name, ds in datasets.items():
                    band = ds.read(
                        1, window=window, masked=True
                    )

                    arr = np.asarray(
                        band.astype(np.float32).filled(np.nan),
                        dtype=np.float32,
                    )

                    arrays[name] = arr
                    valid &= np.isfinite(arr)

                    nodata = ds.nodata
                    if nodata is not None and np.isfinite(nodata):
                        valid &= arr != nodata

                    # WorldCover uses zero as NoData.
                    if name == "land_cover":
                        valid &= arr != 0

                output = np.full(
                    (win_height, win_width),
                    OUTPUT_NODATA,
                    dtype=np.float32,
                )

                rows, cols = np.nonzero(valid)

                if len(rows):
                    # Match training feature order exactly.
                    batch = pd.DataFrame({
                        name: arrays[name][rows, cols]
                        for name in FEATURES
                    })

                    # Categorical predictor remains numeric here;
                    # the fitted OneHotEncoder handles it.
                    scores = model.predict_proba(batch)[:, 1]

                    output[rows, cols] = scores.astype(np.float32)
                    predicted_pixels += len(rows)

                skipped_pixels += valid.size - len(rows)

                dst.write(output, 1, window=window)
                processed_windows += 1

            print(
                f"Processed {processed_windows:,}/{total_windows:,} "
                f"windows | valid pixels predicted: {predicted_pixels:,}"
            )

    print("\n" + "=" * 65)
    print("PREDICTION COMPLETE")
    print("=" * 65)
    print(f"Output: {OUTPUT_RASTER}")
    print(f"Predicted pixels: {predicted_pixels:,}")
    print(f"Pixels left as NoData: {skipped_pixels:,}")
    print("Output values represent model scores, not calibrated probabilities.")

finally:
    for ds in datasets.values():
        ds.close()

    if "reference" in locals():
        reference.close()
