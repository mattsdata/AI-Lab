# ailab-geo

*Author: **Mathew Varghese** · Version 1.0*

Terrain analysis, landform change and shape measurement on georeferenced rasters, for any
domain that reads the land: heritage, flood risk, agriculture, infrastructure. NumPy only.

```python
from ailab_geo import diffuse, hillshade, label_regions, local_relief, measure, read_geotiff

dem = read_geotiff("lidar_dtm.tif")  # optional rasterio
relief = local_relief(dem.values, radius=15)  # small features, hills removed
labels, n = label_regions(relief > 0.3)
for region in measure(labels):
    print(region.area_cells, region.outline_rectangularity, region.fill_ratio)
in_1000_years = diffuse(dem.values, dem.cell_size, kappa=0.005, years=1000)
```

## Components

| Module | What it does |
|---|---|
| `grid` | `GeoGrid` (values + origin + cell size + CRS, pixel <-> map), `read_geotiff`, `haversine_km` |
| `terrain` | `slope_degrees`, `aspect_degrees`, `hillshade`, `local_relief` (Local Relief Model), `diffuse` (hillslope erosion) |
| `regions` | `label_regions` (4/8-connected), `measure`: area, centroid, orientation, elongation, rectangularity, fill ratio, outline rectangularity |

## Design decisions

- **Tested against known geometry:** tilted planes for slope and aspect, flat ground for
  hillshade, the London-Paris distance, volume conservation for erosion, rectangles, discs and
  rings for shape measures (including the pixel-disc effect: pi/4 only in the continuous limit).
- **Local Relief Model with a summed-area table:** O(1) per cell for any radius, NaN-aware.
- **Erosion as diffusion:** the standard soil-creep model with automatic stable time steps and
  closed boundaries; run a range of diffusivity values, because the right one depends on soil
  and climate.
- **Shape tells human from natural:** ring-like features (low fill ratio) with rectangular
  outlines are typical of enclosures; natural mounds are solid and rounded.

## Development

```bash
cd packages/geo
pip install -e ".[dev]"
pytest && ruff check . && ruff format --check . && mypy src
```
