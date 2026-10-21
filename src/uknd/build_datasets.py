from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------
# Project defaults. Override paths with command-line arguments if needed.
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RASTER_DIR = Path(r"E:\DS\uknd_raster_backup")
DEFAULT_FEATURES = DEFAULT_RASTER_DIR / "ml_aligned" / "landslide_features.csv"
DEFAULT_INVENTORY = DEFAULT_RASTER_DIR / "uttarakhand_landslides_points.geojson"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "processed" / "BG_A"

CRS = "EPSG:32644"
SEED = 42
BLOCK_SIZE_M = 10_000
WINDOW_SIZE = 512
# A reservoir is retained for each spatial block. Increase this if the
# final allocation asks a block for more candidates than its reservoir.
RESERVOIR_PER_BLOCK = 2_000

# Keys here are output feature names. Paths point to the final raster stack.
RASTER_FILES = {
    "elevation": Path("dem/tiles/uk_dem_glo30_utm44n_clipped.tif"),
    "slope": Path("dem/tiles/uk_slope_cleaned.tif"),
    "profile_curvature": Path("terrain/uk_profile_curvature.tif"),
    "plan_curvature": Path("terrain/uk_plan_curvature.tif"),
    "land_cover": Path("ml_aligned/uk_worldcover_dem_grid.tif"),
    "aspect": Path("ml_aligned/uk_aspect_dem_grid.tif"),
    "distance_to_road": Path("ROADS/uk_distance_to_road_final.tif"),
    "distance_to_river": Path("RIVERS/uk_distance_to_river_aligned_v2.tif"),
}


def log(message: str) -> None:
    print(message, flush=True)


def load_inventory(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Inventory not found: {path}")

    inventory = gpd.read_file(path)
    if inventory.crs is None:
        raise ValueError(
            "Inventory CRS is missing. Do not assign one by assumption; "
            "verify the source CRS first."
        )

    inventory = inventory.to_crs(CRS)
    inventory = inventory[
        inventory.geometry.notna() & ~inventory.geometry.is_empty
    ].copy()
    inventory = inventory[inventory.geometry.geom_type == "Point"].copy()

    if inventory.empty:
        raise ValueError("No valid Point geometries found in inventory.")

    return inventory


def load_complete_positives(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    df = pd.read_csv(path)

    predictors = list(RASTER_FILES)
    required = set(predictors) | {"target", "all_predictors_valid"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Feature CSV is missing required columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    validity = df["all_predictors_valid"].astype(str).str.strip().str.lower()
    valid = validity.isin({"true", "1", "yes"})
    valid &= df[predictors].notna().all(axis=1)
    valid &= pd.to_numeric(df["target"], errors="coerce").eq(1)

    positives = df.loc[valid].copy()
    if positives.empty:
        raise ValueError("No complete positive samples found.")

    # Convert predictors to numeric and remove any non-finite values.
    for column in predictors:
        positives[column] = pd.to_numeric(positives[column], errors="coerce")
    finite = np.isfinite(positives[predictors].to_numpy(dtype=float)).all(axis=1)
    positives = positives.loc[finite].copy()

    if positives.empty:
        raise ValueError("No finite complete positive samples remain.")

    log(f"Complete positive samples: {len(positives):,}")
    return positives


def open_and_validate_rasters(raster_dir: Path):
    datasets = {}
    reference = None

    for name, relative_path in RASTER_FILES.items():
        path = raster_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing raster for {name}: {path}")

        ds = rasterio.open(path)
        if ds.crs is None or ds.crs.to_string() != CRS:
            ds.close()
            raise ValueError(
                f"{name} CRS is {ds.crs}; expected {CRS}. "
                "Reproject/resample correctly before sampling."
            )

        if reference is None:
            reference = ds
        else:
            if (
                ds.width != reference.width
                or ds.height != reference.height
                or not ds.transform.almost_equals(reference.transform)
                or ds.crs != reference.crs
            ):
                ds.close()
                raise ValueError(
                    f"{name} does not exactly align with the reference grid. "
                    "Align the raster before building BG-A."
                )

        datasets[name] = ds

    log(
        f"Validated {len(datasets)} rasters on "
        f"{reference.width:,} x {reference.height:,} grid ({CRS})."
    )
    return datasets, reference


def reservoir_add(reservoir: list, seen: int, item: tuple, capacity: int, rng):
    """Classic reservoir sampling: uniform sample from items seen so far."""
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        j = int(rng.integers(0, seen))
        if j < capacity:
            reservoir[j] = item


def scan_valid_cells(
    datasets: dict,
    reference,
    inventory: gpd.GeoDataFrame,
    buffer_m: float,
    rng: np.random.Generator,
    dry_run: bool,
):
    """Scan raster windows, keeping a uniform reservoir within each 10 km block.

    This avoids point-by-point Rasterio sampling. Counts are exact for the
    scanned grid; stored candidates are a reproducible per-block reservoir.
    """
    inventory_xy = np.column_stack(
        (inventory.geometry.x.to_numpy(), inventory.geometry.y.to_numpy())
    )
    tree = cKDTree(inventory_xy)

    reservoirs = defaultdict(list)
    eligible_counts = defaultdict(int)
    total_cells_checked = 0
    total_valid_after_buffer = 0
    nodata_values = {name: ds.nodata for name, ds in datasets.items()}

    for row_off in range(0, reference.height, WINDOW_SIZE):
        height = min(WINDOW_SIZE, reference.height - row_off)

        for col_off in range(0, reference.width, WINDOW_SIZE):
            width = min(WINDOW_SIZE, reference.width - col_off)
            window = Window(col_off, row_off, width, height)

            arrays = {}
            valid = np.ones((height, width), dtype=bool)

            for name, ds in datasets.items():
                band = ds.read(1, window=window, masked=True)
                arr = band.astype(np.float32).filled(np.nan)
                arr = np.asarray(arr, dtype=np.float32)
                arrays[name] = arr
                valid &= np.isfinite(arr)

                nodata = nodata_values[name]
                if nodata is not None and np.isfinite(nodata):
                    valid &= arr != nodata

                # This project's WorldCover raster uses 0 as NoData.
                if name == "land_cover":
                    valid &= arr != 0

            local_rows, local_cols = np.nonzero(valid)
            if len(local_rows) == 0:
                total_cells_checked += height * width
                continue

            global_rows = local_rows + row_off
            global_cols = local_cols + col_off
            xs, ys = rasterio.transform.xy(
                reference.transform,
                global_rows,
                global_cols,
                offset="center",
            )
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)

            distances, _ = tree.query(np.column_stack((xs, ys)), k=1)
            keep = distances >= buffer_m

            if np.any(keep):
                global_rows = global_rows[keep]
                global_cols = global_cols[keep]
                xs = xs[keep]
                ys = ys[keep]
                local_rows = local_rows[keep]
                local_cols = local_cols[keep]

                # Spatial blocks are based on projected coordinates.
                block_x = np.floor(xs / BLOCK_SIZE_M).astype(np.int32)
                block_y = np.floor(ys / BLOCK_SIZE_M).astype(np.int32)

                for i in range(len(xs)):
                    key = (int(block_x[i]), int(block_y[i]))
                    eligible_counts[key] += 1
                    total_valid_after_buffer += 1

                    # Save the cell index and predictor values in the reservoir.
                    values = tuple(
                        float(arrays[name][local_rows[i], local_cols[i]])
                        for name in RASTER_FILES
                    )
                    item = (
                        int(global_rows[i]),
                        int(global_cols[i]),
                        float(xs[i]),
                        float(ys[i]),
                        *values,
                    )
                    reservoir_add(
                        reservoirs[key],
                        eligible_counts[key],
                        item,
                        RESERVOIR_PER_BLOCK,
                        rng,
                    )

            total_cells_checked += height * width

        log(
            f"Scanned through row {row_off + height:,}/{reference.height:,}; "
            f"eligible cells so far: {total_valid_after_buffer:,}"
        )

    log(f"Grid cells scanned: {total_cells_checked:,}")
    log(f"Eligible cells outside {buffer_m:g} m buffer: {total_valid_after_buffer:,}")
    log(f"Spatial blocks with eligible cells: {len(eligible_counts):,}")

    if dry_run:
        return None, eligible_counts, reservoirs

    return reservoirs, eligible_counts, reservoirs


def select_spatially_stratified(
    reservoirs: dict,
    eligible_counts: dict,
    target_count: int,
    rng: np.random.Generator,
):
    """Allocate sample quotas proportional to exact eligible-cell counts."""
    available = sum(len(items) for items in reservoirs.values())
    if available < target_count:
        raise ValueError(
            f"Only {available:,} retained candidate cells across block "
            f"reservoirs, fewer than target {target_count:,}. Increase "
            "RESERVOIR_PER_BLOCK and rerun."
        )

    total_eligible = sum(eligible_counts.values())
    if total_eligible < target_count:
        raise ValueError(
            f"Only {total_eligible:,} eligible cells for {target_count:,} "
            "requested samples."
        )

    blocks = list(eligible_counts)
    exact_quotas = {
        b: target_count * eligible_counts[b] / total_eligible for b in blocks
    }
    quotas = {b: int(np.floor(exact_quotas[b])) for b in blocks}
    remaining = target_count - sum(quotas.values())

    order = sorted(
        blocks,
        key=lambda b: exact_quotas[b] - quotas[b],
        reverse=True,
    )
    for b in order:
        if remaining <= 0:
            break
        quotas[b] += 1
        remaining -= 1

    # If a block's quota exceeds its retained reservoir, cap it and
    # redistribute the shortfall to blocks with spare reservoir capacity.
    for b in blocks:
        quotas[b] = min(quotas[b], len(reservoirs[b]))

    shortfall = target_count - sum(quotas.values())
    while shortfall > 0:
        eligible_blocks = [
            b for b in blocks if quotas[b] < len(reservoirs[b])
        ]
        if not eligible_blocks:
            raise ValueError(
                "Reservoir capacity cannot satisfy the requested sample "
                "count. Increase RESERVOIR_PER_BLOCK and rerun."
            )

        # Prefer blocks with the largest remaining proportional deficit.
        eligible_blocks.sort(
            key=lambda b: (
                exact_quotas[b] - quotas[b],
                len(reservoirs[b]) - quotas[b],
            ),
            reverse=True,
        )
        for b in eligible_blocks:
            if shortfall <= 0:
                break
            quotas[b] += 1
            shortfall -= 1

    selected = []
    for b in blocks:
        quota = quotas[b]
        if quota == 0:
            continue
        choices = rng.choice(len(reservoirs[b]), size=quota, replace=False)
        selected.extend(reservoirs[b][int(i)] for i in choices)

    if len(selected) != target_count:
        raise RuntimeError(
            f"Selected {len(selected)} samples, expected {target_count}."
        )

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Build BG-A background samples and combined modelling CSV."
    )
    parser.add_argument("--raster-dir", type=Path, default=DEFAULT_RASTER_DIR)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buffer", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--compare-buffers",
        action="store_true",
        help="Audit 30, 60 and 100 m buffers; do not write datasets.",
    )
    args = parser.parse_args()

    if args.buffer < 0:
        parser.error("--buffer must be non-negative.")

    log("=" * 60)
    log("BG-A BACKGROUND DATASET BUILDER")
    log("=" * 60)
    log(f"Project: {PROJECT_DIR}")
    log(f"Raster directory: {args.raster_dir}")
    log(f"Features CSV: {args.features}")
    log(f"Inventory: {args.inventory}")
    log(f"Output directory: {args.output_dir}")

    positives = load_complete_positives(args.features)
    inventory = load_inventory(args.inventory)
    datasets, reference = open_and_validate_rasters(args.raster_dir)

    try:
        buffers = [30.0, 60.0, 100.0] if args.compare_buffers else [args.buffer]

        if args.compare_buffers:
            # Use the same seed per buffer for reproducibility.
            for buffer_m in buffers:
                log(f"\nTesting exclusion buffer: {buffer_m:g} m")
                scan_valid_cells(
                    datasets, reference, inventory, buffer_m,
                    np.random.default_rng(args.seed), dry_run=True
                )
            log("\nBuffer comparison complete. No output files written.")
            return

        if args.dry_run:
            log(f"\nDry run with buffer: {args.buffer:g} m")
            scan_valid_cells(
                datasets, reference, inventory, args.buffer,
                np.random.default_rng(args.seed), dry_run=True
            )
            log("Dry run complete. No output files written.")
            return

        log(f"\nGenerating BG-A using {args.buffer:g} m exclusion buffer.")
        reservoirs, counts, _ = scan_valid_cells(
            datasets, reference, inventory, args.buffer,
            np.random.default_rng(args.seed), dry_run=False
        )

        selected = select_spatially_stratified(
            reservoirs, counts, len(positives),
            np.random.default_rng(args.seed)
        )

        columns = [
            "row", "col", "x_utm44n", "y_utm44n", *RASTER_FILES.keys()
        ]
        background = pd.DataFrame(selected, columns=columns)
        background.insert(
            0, "background_id",
            [f"BG-A-{i:06d}" for i in range(1, len(background) + 1)]
        )
        background["target"] = 0
        background["sample_type"] = "background_unrecorded"
        background["sampling_method"] = "window_scanned_spatially_stratified"
        background["buffer_m"] = args.buffer
        background["random_seed"] = args.seed
        background["block_x"] = np.floor(
            background["x_utm44n"] / BLOCK_SIZE_M
        ).astype(int)
        background["block_y"] = np.floor(
            background["y_utm44n"] / BLOCK_SIZE_M
        ).astype(int)
        background = background.drop(columns=["row", "col"])

        args.output_dir.mkdir(parents=True, exist_ok=True)
        bg_path = args.output_dir / "background_samples.csv"
        combined_path = args.output_dir / "combined_training.csv"
        metadata_path = args.output_dir / "sampling_metadata.json"
        counts_path = args.output_dir / "eligible_cells_by_block.csv"

        background.to_csv(bg_path, index=False)

        positives_out = positives.copy()
        positives_out["sample_type"] = "recorded_landslide"
        positives_out["sampling_method"] = "inventory"
        positives_out["buffer_m"] = np.nan
        positives_out["random_seed"] = args.seed

        # Harmonize label dtype and retain positive feature columns.
        positives_out["target"] = 1
        combined = pd.concat(
            [positives_out, background],
            ignore_index=True,
            sort=False,
        )
        combined.to_csv(combined_path, index=False)

        block_counts = pd.DataFrame([
            {
                "block_x": block[0],
                "block_y": block[1],
                "eligible_cell_count": count,
                "reservoir_candidate_count": len(reservoirs[block]),
            }
            for block, count in counts.items()
        ])
        block_counts.to_csv(counts_path, index=False)

        metadata = {
            "sampling_design": "BG-A",
            "background_semantics": (
                "No recorded landslide at sampled location; not verified "
                "stable terrain and not a confirmed negative."
            ),
            "positive_count": int(len(positives_out)),
            "background_count": int(len(background)),
            "combined_count": int(len(combined)),
            "buffer_m": args.buffer,
            "spatial_block_size_m": BLOCK_SIZE_M,
            "reservoir_per_block": RESERVOIR_PER_BLOCK,
            "random_seed": args.seed,
            "crs": CRS,
            "predictors": list(RASTER_FILES),
            "raster_paths": {
                name: str(args.raster_dir / rel)
                for name, rel in RASTER_FILES.items()
            },
            "inventory_path": str(args.inventory),
            "features_path": str(args.features),
            "notes": [
                "Original inventory and feature CSV are not modified.",
                "All inventory points, including incomplete records, are used "
                "for the exclusion buffer.",
                "Only complete positive feature records are included in the "
                "combined modelling CSV.",
                "Spatial block quotas are proportional to eligible cell "
                "counts, subject to per-block reservoir capacity."
            ],
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        log("\nBG-A generation complete.")
        log(f"Background samples: {bg_path}")
        log(f"Combined dataset:    {combined_path}")
        log(f"Block diagnostics:   {counts_path}")
        log(f"Metadata:            {metadata_path}")
        log(f"Positive samples:    {len(positives_out):,}")
        log(f"Background samples:  {len(background):,}")
        log(f"Combined rows:       {len(combined):,}")

    finally:
        for ds in datasets.values():
            ds.close()


if __name__ == "__main__":
    main()
