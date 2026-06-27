# AGENTS.md — Project Context

## Dataset Comparison: Bradbury vs BDAPPV

### Bradbury (2016)

| Property | Value |
|----------|-------|
| **Total polygons** | 19,435 (13,800 high-quality, J≥0.5) |
| **Unique images** | 463 |
| **Image size** | 5000×5000 px GeoTIFF (USGS orthoimagery) |
| **GSD** | 0.3 m/pixel |
| **Annotation format** | CSV polygon vertices (pixel coords), 4–146 verts/poly |
| **Multi-instance** | Many panels per image (full scene) |
| **Metadata** | Minimal (Jaccard index, city, area) |
| **Coverage** | 4 CA cities: Fresno (14,712), Stockton (2,546), Oxnard (1,595), Modesto (582) |
| **Preprocessing** | Centroid-based 400×400 tile extraction from full orthoimages |
| **License** | CC0 |
| **Reference** | Bradbury et al. "Distributed solar photovoltaic array location and extent dataset" (Scientific Data, 2016) |

### BDAPPV (Kasmi 2023)

| Property | Value |
|----------|-------|
| **Total masks** | 20,988 |
| **Installation records** | 28,408 |
| **Tile size** | 400×400 px PNG masks |
| **GSD** | 0.1–0.2 m/pixel (variable, two aerial sources) |
| **Annotation format** | PNG pixel masks (uint8, 0/255) |
| **Multi-instance** | One panel per tile (pre-isolated) |
| **Metadata** | Rich (kWp, tilt, azimuth, surface, city, self-consumption, inverter/array specs, installation date) |
| **Coverage** | All of France (départements), 2 sources: Google + IGN |
| **Preprocessing** | Connected-components labeling on binary masks → instance masks |
| **License** | CC-BY 4.0 |
| **Reference** | Kasmi et al. "A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays" (Scientific Data, 2023) |

### Key Differences Summary

1. **Annotation format**: Bradbury = vector polygons in CSV; BDAPPV = raster masks in PNG
2. **Image size**: Bradbury = 5000² px full scenes; BDAPPV = 400² px pre-cropped tiles
3. **Multi-instance**: Bradbury = many panels per image (need centroid extraction); BDAPPV = one installation per tile (need connected components)
4. **Metadata**: Bradbury = minimal; BDAPPV = extensive installation attributes
5. **Coverage**: Bradbury = 4 CA cities; BDAPPV = all France
6. **Resolution**: Bradbury = 0.3m; BDAPPV = 0.1–0.2m (higher)
7. **Preprocessing goal**: Bradbury = tile extraction from full scenes; BDAPPV = instance isolation from masks
