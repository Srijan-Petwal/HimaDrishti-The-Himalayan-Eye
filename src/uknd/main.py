
from pathlib import Path
import requests

OUT = Path("data/raw/dem/tiles")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"

# Candidate 1-degree tiles: 77–80 E, 28–31 N.
# The actual state boundary will be used for clipping later.
tiles = [
    (lat, lon)
    for lat in range(28, 32)
    for lon in range(77, 81)
]

session = requests.Session()
downloaded = []
missing = []

for lat, lon in tiles:
    tile = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
    url = f"{BASE}/{tile}/{tile}.tif"
    dest = OUT / f"{tile}.tif"

    if dest.exists() and dest.stat().st_size > 0:
        print("Already exists:", dest.name)
        downloaded.append(dest)
        continue

    try:
        with session.get(
            url, stream=True, timeout=(15, 120)
        ) as response:
            if response.status_code == 404:
                print("Not available:", tile)
                missing.append(tile)
                continue

            response.raise_for_status()
            temp = dest.with_suffix(".tif.part")

            with temp.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if temp.stat().st_size < 100_000:
            temp.unlink(missing_ok=True)
            print("Unexpectedly small response:", tile)
            missing.append(tile)
            continue

        temp.replace(dest)
        downloaded.append(dest)
        print("Downloaded:", dest.name)

    except requests.RequestException as exc:
        print("Download failed:", tile, exc)
        missing.append(tile)

print("\nSuccessful tiles:", len(downloaded))
print("Missing or failed tiles:", len(missing))
print("Tile folder:", OUT.resolve())

if missing:
    print("\nCheck these tiles individually or use GLO-90 as a fallback:")
    for name in missing:
        print(" -", name)
