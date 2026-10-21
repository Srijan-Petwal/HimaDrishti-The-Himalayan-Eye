
from pathlib import Path

import numpy as np
import rasterio

RASTER_PATH = Path(
    r"E:\DS\uknd\outputs\susceptibility"
    r"\uknd_xgboost_susceptibility.tif"
)

OUTPUT_DIR = Path(r"E:\DS\uknd\outputs\susceptibility")
REPORT_PATH = OUTPUT_DIR / "score_distribution.txt"

PERCENTILES = [0, 1, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99, 100]
THRESHOLDS = [0.2, 0.4, 0.6, 0.8]


def main():
    if not RASTER_PATH.exists():
        raise FileNotFoundError(RASTER_PATH)

    # First pass: count valid cells.
    valid_count = 0

    with rasterio.open(RASTER_PATH) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = np.isfinite(arr)

            if src.nodata is not None:
                valid &= arr != src.nodata

            valid_count += int(valid.sum())

    if valid_count == 0:
        raise ValueError("No valid raster scores found.")

    # Exact percentiles need all valid scores in memory.
    # A float32 array uses approximately 4 bytes per valid cell.
    print(f"Valid pixels: {valid_count:,}")
    print(f"Estimated score-array memory: {valid_count * 4 / 1e9:.2f} GB")

    scores = np.empty(valid_count, dtype=np.float32)
    pos = 0

    with rasterio.open(RASTER_PATH) as src:
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = np.isfinite(arr)

            if src.nodata is not None:
                valid &= arr != src.nodata

            values = arr[valid]
            scores[pos:pos + len(values)] = values
            pos += len(values)

    scores = scores[:pos]

    percentiles = np.percentile(scores, PERCENTILES)

    lines = [
        "XGBoost susceptibility score distribution",
        f"Raster: {RASTER_PATH}",
        f"Valid pixels: {len(scores):,}",
        "",
        "Percentiles:",
    ]

    print("\nPercentiles:")
    for p, value in zip(PERCENTILES, percentiles):
        line = f"P{p:>3}: {value:.6f}"
        print(line)
        lines.append(line)

    lines.append("")
    lines.append("Threshold exceedance:")
    print("\nThreshold exceedance:")

    for threshold in THRESHOLDS:
        count = int(np.count_nonzero(scores >= threshold))
        fraction = count / len(scores) * 100

        line = (
            f"Score >= {threshold:.1f}: "
            f"{count:,} pixels ({fraction:.2f}%)"
        )
        print(line)
        lines.append(line)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
