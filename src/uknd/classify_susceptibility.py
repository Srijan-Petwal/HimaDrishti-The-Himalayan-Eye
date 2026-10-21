
from pathlib import Path

import numpy as np
import rasterio

INPUT_RASTER = Path(
    r"E:\DS\uknd\outputs\susceptibility"
    r"\uknd_xgboost_susceptibility.tif"
)

OUTPUT_RASTER = Path(
    r"E:\DS\uknd\outputs\susceptibility"
    r"\uknd_susceptibility_5class.tif"
)

# Approximate percentile breaks from the raster analysis.
# Class codes: 1=Very low, 2=Low, 3=Moderate,
#              4=High, 5=Very high.
BREAKS = [
    0.005305,
    0.020048,
    0.116785,
    0.346057,
]

NODATA = 0
WINDOW_SIZE = 512


def main():
    if not INPUT_RASTER.exists():
        raise FileNotFoundError(INPUT_RASTER)

    with rasterio.open(INPUT_RASTER) as src:
        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=NODATA,
            compress="deflate",
            predictor=1,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="IF_SAFER",
        )

        counts = np.zeros(6, dtype=np.int64)
        nodata_count = 0

        with rasterio.open(OUTPUT_RASTER, "w", **profile) as dst:
            for row in range(0, src.height, WINDOW_SIZE):
                height = min(WINDOW_SIZE, src.height - row)

                for col in range(0, src.width, WINDOW_SIZE):
                    width = min(WINDOW_SIZE, src.width - col)

                    window = rasterio.windows.Window(
                        col, row, width, height
                    )

                    scores = src.read(1, window=window)
                    valid = np.isfinite(scores)

                    if src.nodata is not None:
                        valid &= scores != src.nodata

                    classified = np.zeros(
                        scores.shape, dtype=np.uint8
                    )

                    values = scores[valid]

                    # Right-inclusive boundaries ensure a score
                    # equal to a break stays in the lower class.
                    classes = np.digitize(
                        values,
                        BREAKS,
                        right=True,
                    ) + 1

                    classified[valid] = classes.astype(np.uint8)

                    for class_id in range(1, 6):
                        counts[class_id] += np.count_nonzero(
                            classified == class_id
                        )

                    nodata_count += np.count_nonzero(~valid)
                    dst.write(classified, 1, window=window)

    print("Classification complete.")
    print(f"Output: {OUTPUT_RASTER}")
    print(f"NoData cells: {nodata_count:,}")

    names = {
        1: "Very low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very high",
    }

    total_valid = counts[1:].sum()

    for class_id in range(1, 6):
        count = counts[class_id]
        percentage = 100 * count / total_valid if total_valid else 0
        print(
            f"Class {class_id} ({names[class_id]}): "
            f"{count:,} pixels ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    main()
