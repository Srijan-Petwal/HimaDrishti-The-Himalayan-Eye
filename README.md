## Datasets

### Source Datasets

| Dataset | Description |
|---|---|
| **Uttarakhand Landslide Inventory** | GeoJSON point inventory digitized from a PDF, representing recorded landslide locations. |
| **Uttarakhand Administrative Boundary** | State boundary obtained from [geoBoundaries](https://www.geoboundaries.org/), used to define the study area. |
| **Copernicus GLO-30 DEM** | 30 m-class digital elevation model used to derive terrain predictors. |
| **ESA WorldCover 2021 v200** | 10 m land-cover classification assembled from four tiles covering the study area. |
| **HydroRIVERS v1.0** | River network dataset used to calculate proximity to rivers. |

### Processed Datasets

| Dataset | Description |
|---|---|
| `uk_landslides_points_final.gpkg` | Landslide inventory filtered to points intersecting the Uttarakhand boundary. |
| `uk_boundary_utm44n.gpkg` | Reprojected state boundary in **EPSG:32644 (WGS 84 / UTM zone 44N)**. |
| `uk_dem_glo30_utm44n_clipped.tif` | DEM reprojected to UTM zone 44N and clipped to the study area. |
| `uk_slope_cleaned` | Slope raster derived from the DEM, expressed in degrees. |
| `uk_aspect` | Aspect raster derived from the DEM, representing slope orientation in degrees. |
| `uk_worldcover_aligned_nodata.tif` | WorldCover mosaic clipped to the study area, with NoData handling and alignment to the DEM grid using nearest-neighbour resampling. |
| `uk_rivers_extracted_utm44n.gpkg` | HydroRIVERS network reprojected and filtered to the study area. |
| `uk_rivers_raster.tif` | Binary raster representation of the extracted river network. |
| `uk_distance_to_river_clipped` | Raster of distance to the nearest river, in metres, clipped to the study area. |

### Current Predictors

- **Elevation:** Copernicus GLO-30 DEM.
- **Slope:** Derived from the DEM.
- **Aspect:** Derived from the DEM.
- **Land Cover:** ESA WorldCover 2021 v200.
- **Distance to Rivers:** Derived from HydroRIVERS.

The processed rasters use a common projected grid where applicable, enabling spatial alignment for subsequent modelling.

> [!NOTE]
**Caveat** Commiting Large source and processed datasets did not seem to make sense and hence are excluded from version control. Refer to the original data providers and preprocessing workflow to reproduce them.