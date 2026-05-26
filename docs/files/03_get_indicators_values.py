"""
AGB & Tree Cover Raster Statistics by Area of Interest
=======================================================

Calculates mean, min, and max for rasters masked to shapefile boundaries.
Statistics are computed over all valid pixels within the entire shapefile area
(not per polygon). Also reports the area of each AOI shapefile in hectares.

INPUTS:
-------
- AGB rasters: GEDI (2023)
- MOD44B multi-band raster: last band (most recent year) used
- Two shapefiles as areas of interest

OUTPUTS:
--------
- Printed summary table with mean, min, max per raster/AOI combination
- AOI area in hectares

Last Updated: 2026-05-08
"""

import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
import os
from datetime import datetime


# =============================================================================
# Configuration
# =============================================================================

# Each entry: "label" -> {"path": ..., "band": int or -1 for last, "units": str}
RASTERS = {
    "GEDI AGB 2023": {
        "path":  r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\7_AGB\GEDI\GEDI_AGB_2023_raw.tif",
        "band":  1,
        "units": "Mg/ha",
    },
    "MOD44B Tree Cover (last yr)": {
        "path":  r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\response\MOD44B_TreeCover_2014-2023.tif",
        "band":  -1,   # -1 = last band
        "units": "%",
    },
}

AREAS_OF_INTEREST = {
    "Project Extent - Condition at Project Start": r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\Project_Extent\project_extent_v2.shp",
    "Core Zone - Reference Value": r"A:\TGC-01201 Amigos de Calakmul - Mexico\1. GIS\2. Working Files\28. Additional Information\Core_Zone.shp",
}

OUTPUT_DIR = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\outputs"
OUTPUT_TXT  = os.path.join(OUTPUT_DIR, "indicators_values.txt")


# =============================================================================
# Helpers
# =============================================================================

def get_shapefile_area_ha(shp_path):
    """
    Return the total area of all features in a shapefile in hectares.
    Reprojects to EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0, equal-area) for
    accurate area calculation regardless of the original CRS.
    """
    gdf = gpd.read_file(shp_path)
    gdf_ea = gdf.to_crs(epsg=6933)
    area_ha = gdf_ea.geometry.area.sum() / 10_000
    return area_ha


def get_raster_stats_by_aoi(raster_path, band_index, units, shp_path, raster_name, aoi_name):
    """
    Extract mean, min, and max from one band of a raster masked to a shapefile.

    Args:
        raster_path : Path to the input raster (.tif)
        band_index  : 1-based band number, or -1 to use the last band
        units       : String label for the units (e.g. "Mg/ha", "%")
        shp_path    : Path to the shapefile defining the AOI
        raster_name : Display label for the raster
        aoi_name    : Display label for the AOI

    Returns:
        dict with keys 'mean', 'min', 'max', 'count', or None if no valid pixels
    """
    print(f"\n  Raster : {raster_name}")
    print(f"  AOI    : {aoi_name}")

    with rasterio.open(raster_path) as src:
        n_bands = src.count
        band_num = n_bands if band_index == -1 else band_index
        print(f"  Band   : {band_num}/{n_bands}")

        raster_data = src.read(band_num).astype("float32")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = src.shape

    # Mask nodata and NaN
    invalid = np.isnan(raster_data)
    if nodata is not None:
        invalid |= raster_data == nodata

    # Reproject shapefile to raster CRS if needed
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Burn all geometries into a single boolean mask (True = inside AOI)
    aoi_mask = features.geometry_mask(
        gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True,
    )

    valid_pixels = raster_data[aoi_mask & ~invalid]

    if valid_pixels.size == 0:
        print("  WARNING: No valid pixels found within AOI.")
        return None

    stats = {
        "mean":  float(np.mean(valid_pixels)),
        "min":   float(np.min(valid_pixels)),
        "max":   float(np.max(valid_pixels)),
        "count": int(valid_pixels.size),
        "units": units,
    }

    print(f"  Valid pixels : {stats['count']:,}")
    print(f"  Mean  : {stats['mean']:.4f} {units}")
    print(f"  Min   : {stats['min']:.4f} {units}")
    print(f"  Max   : {stats['max']:.4f} {units}")

    return stats


# =============================================================================
# Main Workflow
# =============================================================================

def run_agb_statistics():
    """
    Run statistics for all raster / AOI combinations, print a summary, and
    save two tables plus the input paths to a text file.
    """
    print("\n" + "=" * 70)
    print("RASTER STATISTICS BY AREA OF INTEREST")
    print("=" * 70)

    areas = {}
    results = {}

    for aoi_name, shp_path in AREAS_OF_INTEREST.items():
        areas[aoi_name] = get_shapefile_area_ha(shp_path)

    for aoi_name, shp_path in AREAS_OF_INTEREST.items():
        print(f"\n{'─' * 65}")
        print(f"AOI: {aoi_name}")
        print(f"{'─' * 65}")
        results[aoi_name] = {}
        for raster_name, cfg in RASTERS.items():
            stats = get_raster_stats_by_aoi(
                cfg["path"], cfg["band"], cfg["units"],
                shp_path, raster_name, aoi_name,
            )
            results[aoi_name][raster_name] = stats

    # Build output lines (printed to console and written to file)
    lines = []

    lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Inputs section
    lines.append("=" * 65)
    lines.append("INPUTS")
    lines.append("=" * 65)
    lines.append("Rasters:")
    for raster_name, cfg in RASTERS.items():
        band_label = "last" if cfg["band"] == -1 else str(cfg["band"])
        lines.append(f"  {raster_name}")
        lines.append(f"    path : {cfg['path']}")
        lines.append(f"    band : {band_label}  |  units : {cfg['units']}")
    lines.append("")
    lines.append("Areas of interest:")
    for aoi_name, shp_path in AREAS_OF_INTEREST.items():
        lines.append(f"  {aoi_name}")
        lines.append(f"    path : {shp_path}")
    lines.append("")

    # Table 1 — AOI areas (dynamic column widths)
    w_aoi_t1 = max(len("AOI"), max(len(n) for n in areas))
    w_area   = max(len("Area (ha)"), 12)
    sep_len_t1 = w_aoi_t1 + 2 + w_area

    lines.append("=" * sep_len_t1)
    lines.append("TABLE 1 — AOI AREAS")
    lines.append("=" * sep_len_t1)
    lines.append(f"{'AOI':<{w_aoi_t1}}  {'Area (ha)':>{w_area}}")
    lines.append("-" * sep_len_t1)
    for aoi_name, area_ha in areas.items():
        lines.append(f"{aoi_name:<{w_aoi_t1}}  {area_ha:>{w_area},.1f}")
    lines.append("=" * sep_len_t1)
    lines.append("")

    # Table 2 — Raster statistics (dynamic column widths)
    w_aoi    = max(len("AOI"),    max(len(n) for n in results))
    w_raster = max(len("Raster"), max(len(n) for n in RASTERS))
    w_num    = 10   # width for Mean / Min / Max values
    sep_len  = w_aoi + 2 + w_raster + 2 + w_num * 3 + 2 + 5   # +5 for Units label

    lines.append("=" * sep_len)
    lines.append("TABLE 2 — RASTER STATISTICS")
    lines.append("=" * sep_len)
    lines.append(
        f"{'AOI':<{w_aoi}}  {'Raster':<{w_raster}}  "
        f"{'Mean':>{w_num}} {'Min':>{w_num}} {'Max':>{w_num}}  Units"
    )
    lines.append("-" * sep_len)
    for aoi_name, raster_results in results.items():
        for raster_name, stats in raster_results.items():
            if stats:
                lines.append(
                    f"{aoi_name:<{w_aoi}}  {raster_name:<{w_raster}}  "
                    f"{stats['mean']:>{w_num}.2f} {stats['min']:>{w_num}.2f} {stats['max']:>{w_num}.2f}  {stats['units']}"
                )
            else:
                lines.append(
                    f"{aoi_name:<{w_aoi}}  {raster_name:<{w_raster}}  "
                    f"{'N/A':>{w_num}} {'N/A':>{w_num}} {'N/A':>{w_num}}"
                )
    lines.append("=" * sep_len)

    # Print to console
    print()
    for line in lines:
        print(line)

    # Write to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to: {OUTPUT_TXT}")

    return results


if __name__ == "__main__":
    run_agb_statistics()
