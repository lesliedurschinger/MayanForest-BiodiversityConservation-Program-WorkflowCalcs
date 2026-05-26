"""
Download MODIS MOD44B Vegetation Continuous Fields (VCF) 2014–2023
===================================================================

Downloads the MODIS MOD44B annual Vegetation Continuous Fields product for a
given area of interest and stacks all years into a single multi-band GeoTIFF.

Source: Google Earth Engine
- Collection: MODIS/061/MOD44B
- Band: Percent_Tree_Cover (0–100 %; fill value = 200)
- Resolution: 250 m
- Available: yearly, ~day 065 (early March) of each year

INPUTS:
-------
- Shapefile defining the area of interest (AOI)

OUTPUTS:
--------
- Single multi-band GeoTIFF with one band per year (2014–2023),
  each band named tree_cover_{year}. Fill-value pixels (200) are masked.

DEPENDENCIES:
-------------
- Google Earth Engine (ee) — authenticated and initialized
- geemap, geopandas
"""

import ee
import geemap
import geopandas as gpd
import os

# =============================================================================
# Configuration
# =============================================================================

# Path to the AOI shapefile
SHAPEFILE_PATH = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\2_Project_extent\Project_Boundary.shp"

# Output directory
OUTPUT_DIR = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\response"

# Output filename
OUTPUT_FILENAME = "MOD44B_TreeCover_2014-2023.tif"

# Year range
START_YEAR = 2014
END_YEAR = 2023

# Export scale in meters (native resolution is 250 m)
SCALE = 250

# =============================================================================
# Main
# =============================================================================

def download_mod44b_stack(shapefile_path, output_dir, output_filename,
                          start_year=2014, end_year=2023, scale=250):
    """
    Build a multi-band MOD44B Percent_Tree_Cover stack for [start_year, end_year]
    and download it as a single GeoTIFF.

    Each band is named tree_cover_{year} and corresponds to one annual image.
    Fill-value pixels (value == 200, encoding water / no-data) are masked out.

    Parameters
    ----------
    shapefile_path : str
        Path to the AOI shapefile.
    output_dir : str
        Directory where the output GeoTIFF will be saved.
    output_filename : str
        Name of the output multi-band file.
    start_year : int
        First year of the stack (inclusive).
    end_year : int
        Last year of the stack (inclusive).
    scale : int
        Export resolution in meters.
    """
    # Load AOI
    print(f"Loading AOI from: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    aoi = geemap.geopandas_to_ee(gdf_wgs84)

    # Build one band per year and collect them
    print(f"Building MOD44B stack for {start_year}–{end_year}...")
    bands = []

    for year in range(start_year, end_year + 1):
        print(f"  [{year}] Fetching annual image...")
        collection = (
            ee.ImageCollection("MODIS/061/MOD44B")
            .filter(ee.Filter.calendarRange(year, year, "year"))
            .filterBounds(aoi)
            .select("Percent_Tree_Cover")
        )

        image_count = collection.size().getInfo()
        if image_count == 0:
            print(f"  [{year}] WARNING: No image found — band will be skipped.")
            continue

        # MOD44B is annual; take the first (and normally only) image.
        # Mask fill value (200) which encodes water / no-data.
        band = (
            collection.first()
            .updateMask(collection.first().lt(200))
            .rename(f"tree_cover_{year}")
        )
        bands.append(band)
        print(f"  [{year}] Band added: tree_cover_{year}")

    if not bands:
        raise ValueError("No MOD44B images were found for any year in the requested range.")

    # Stack all bands into a single image
    print(f"\nStacking {len(bands)} annual bands into one image...")
    stack = ee.Image.cat(bands).clip(aoi)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Remove existing file to avoid permission errors from locked files
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Removed existing file: {output_path}")
        except PermissionError:
            raise PermissionError(
                f"Cannot overwrite '{output_path}' — the file is open in another application. "
                f"Close it in QGIS / ArcGIS and try again."
            )

    print(f"Downloading stack to: {output_path}  (scale={scale}m)")
    geemap.download_ee_image(
        stack,
        filename=output_path,
        scale=scale,
        region=aoi.geometry(),
        crs="EPSG:4326",
    )

    print(f"\nDownload complete: {output_path}")
    print(f"Bands: {[f'tree_cover_{y}' for y in range(start_year, end_year + 1)]}")
    return output_path


if __name__ == "__main__":
    # Initialize Earth Engine
    try:
        ee.Number(1).getInfo()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="ee-tgcwindows11")

    download_mod44b_stack(
        shapefile_path=SHAPEFILE_PATH,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        start_year=START_YEAR,
        end_year=END_YEAR,
        scale=SCALE,
    )
