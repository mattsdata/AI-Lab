# ailab-heritage

Landscape archaeology from elevation data. Finds **candidate** earthworks (banks, walls,
ditches, platforms) by their shape, records evidence with a conservative status, protects
unexcavated locations, and estimates how features fade over 10, 500 and 1000 years.

Design rules (see [ADR-0007](../../docs/adr/0007-heritage-ethics-and-location-protection.md)):

- Software only proposes **candidates**. Status rises to *supported* with documentary
  evidence and to *confirmed* only with field survey or excavation.
- Public outputs show a **generalised location** (about 5 km) by default.
- Landscape change is always a **range**, from slow to fast soil creep.

```python
from ailab_geo import read_geotiff
from ailab_heritage import detect_candidates, degradation

dem = read_geotiff("lidar_dtm_1m.tif")  # needs ailab-geo[raster]
for c in detect_candidates(dem)[:5]:
    print(c.id, round(c.score, 2), c.interpretation)
```

Install: `pip install "ailab-heritage @ git+https://github.com/mattsdata/ai-lab.git#subdirectory=frameworks/heritage"`

Run the demo: `python examples/find_earthworks.py` (synthetic landscape, or pass a GeoTIFF path).
