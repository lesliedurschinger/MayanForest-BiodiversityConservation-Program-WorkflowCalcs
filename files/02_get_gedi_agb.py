"""
Download GEDI L4A Aboveground Biomass (AGB) for a given year
=============================================================

Downloads the GEDI Level 4A Aboveground Biomass Density product for a given
area of interest, aggregated as an annual mean composite.

Source: Google Earth Engine
- Collection: LARSE/GEDI/GEDI04_A_002_MONTHLY
- Band: agbd (Aboveground Biomass Density, Mg/ha)
- Resolution: ~25m (GEDI footprint)
- Available: April 2019 onwards

INPUTS:
-------
- Shapefile defining the area of interest (AOI)

OUTPUTS:
--------
- GeoTIFF with AGB values (Mg/ha) clipped to the AOI

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

# Path to the AOI shapefile (same as 03_get_esa_agb.py)
SHAPEFILE_PATH = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\2_Project_extent\Project_Boundary.shp"

# Output directory
OUTPUT_DIR = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\7_AGB\GEDI"

# Output filename
OUTPUT_FILENAME = "GEDI_AGB_2023_raw.tif"

# Year of interest (GEDI available from 2019 onwards)
YEAR = 2023

# Export scale in meters (GEDI footprint is ~25m)
SCALE = 25

# =============================================================================
# Main
# =============================================================================

def download_gedi_agb(shapefile_path, output_dir, output_filename, year=2023, scale=25):
    """
    Download GEDI L4A AGB annual mean composite for a given year and AOI.

    Parameters
    ----------
    shapefile_path : str
        Path to the AOI shapefile.
    output_dir : str
        Directory where the output GeoTIFF will be saved.
    output_filename : str
        Name of the output file.
    year : int
        Year for the annual composite. GEDI available from 2019 onwards.
    scale : int
        Export resolution in meters.
    """
    # Load AOI
    print(f"Loading AOI from: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    gdf_wgs84 = gdf.to_crs(epsg=4326)
    aoi = geemap.geopandas_to_ee(gdf_wgs84)

    # Filter GEDI L4A monthly collection to the target year
    print(f"Fetching GEDI L4A AGB monthly composites for {year}...")
    collection = (
        ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filterBounds(aoi)
        .select("agbd")
    )

    image_count = collection.size().getInfo()
    if image_count == 0:
        raise ValueError(
            f"No GEDI L4A images found for year {year} within the AOI. "
            f"GEDI is available from April 2019 onwards."
        )

    print(f"Found {image_count} monthly image(s) for {year} — computing annual mean...")

    # Annual mean composite, masking nodata (agbd fill value is -9999)
    agb_image = (
        collection
        .map(lambda img: img.updateMask(img.gt(0)))
        .mean()
        .rename("agb")
    )

    # Clip to AOI
    agb_clipped = agb_image.clip(aoi)

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

    # Download using geemap (splits into tiles to bypass 50MB limit)
    print(f"Downloading AGB to: {output_path}")
    print(f"Scale: {scale}m")

    geemap.download_ee_image(
        agb_clipped,
        filename=output_path,
        scale=scale,
        region=aoi.geometry(),
        crs="EPSG:4326",
    )

    print(f"Download complete: {output_path}")
    return output_path


if __name__ == "__main__":
    # Initialize Earth Engine
    try:
        ee.Number(1).getInfo()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="ee-tgcwindows11")

    download_gedi_agb(
        shapefile_path=SHAPEFILE_PATH,
        output_dir=OUTPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        year=YEAR,
        scale=SCALE,
    )
