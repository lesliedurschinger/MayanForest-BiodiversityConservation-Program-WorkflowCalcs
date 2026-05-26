"""
Crediting Baseline Calculation - SD VISta Nature Framework v1.0 (Compliant)
===========================================================================

This script calculates the ex-ante crediting baseline for biodiversity conservation
projects using the Habitat Conversion Risk Method.

METHODOLOGY OVERVIEW:
--------------------
The baseline represents the expected ecosystem condition in the absence of the
project (the "without-project scenario"). It is calculated following
SD VISta Nature Framework v1.0, Section A1.2 (Habitat Conversion Risk Method).

Two distinct variables are derived from the probability map:
- HighRiskProb_p (continuous, 0.3-1.0): Used in Step H7 for magnitude of expected loss
- HighRisk_p (binary, 0/1): Used in Step H6 for proportion of project at risk

CALCULATION STEPS (SD VISta Section A1.2):
------------------------------------------
Buffer: Create 10km donut buffer around project zone and save as KML
Step H4: Threshold probability map (ConvProb_p > 0.3 -> keep value, else -> 0)
Step H5: Remove exclusion zones from probability map
Step H6: Calculate proportion of high-risk pixels (w) using binary count
Step H7: Estimate ecosystem condition: I_hat = I(t0) x (1 - HighRiskProb_p)
Step H8: Calculate delta (dI = I_hat - I(t0)) for high-risk pixels
Step H9: Calculate average delta across high-risk pixels in project extent
Step H10: Standardize: dC_pre = (avg_delta x w) / Rv
Step H11: Annualize the baseline: B_pre = dC_pre / tx

Last Updated: 2025-10-06
"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
import numpy as np
from rasterio import features
import fiona


# =============================================================================
# Configuration
# =============================================================================

# --------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------
PROBABILITY_MAP = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\outputs\StepH3-1_probability_of_decay.tif"
EXCLUSION_ZONES = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\excluded_areas\Excluded_all.shp"
PROJECT_EXTENT = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\2_Project_extent\project_extent_v2.shp"
REFERENCE_REGION = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\1_GIS\2_Working Files\4_Reference_region\RR_IUCN_forest_v2.shp"
RASTER_PATH = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\response\MOD44B_TreeCover_2014-2023.tif"

# --------------------------------------------------------------------------
# Output folder
# --------------------------------------------------------------------------
OUTPUT_FOLDER = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\outputs"

# Output paths (constructed from OUTPUT_FOLDER)
BUFFER_OUTPUT_PATH = f"{OUTPUT_FOLDER}\\project_extent_10kBuffer.kml"
HIGH_RISK_PROB_PATH = f"{OUTPUT_FOLDER}\\stepH4_high_risk_probability.tif"
EXCLUSION_MASK_PATH = f"{OUTPUT_FOLDER}\\stepH5_HR_exclusions.tif"
OVERLAP_PROPORTION_PATH = f"{OUTPUT_FOLDER}\\StepH6_overlap-proportion.txt"
ESTIMATED_CONDITION_PATH = f"{OUTPUT_FOLDER}\\stepH7_estimated_ecosystem_condition.tif"
DELTA_HIGH_RISK_PATH = f"{OUTPUT_FOLDER}\\stepH8_delta_high_risk_pixels.tif"
AVERAGE_DELTA_PATH = f"{OUTPUT_FOLDER}\\stepH9_average_delta_high_risk.txt"
STANDARDIZED_CHANGE_PATH = f"{OUTPUT_FOLDER}\\stepH10_deltaPrediction_ex_ante.txt"
CREDITING_BASELINE_PATH = f"{OUTPUT_FOLDER}\\stepH11_annual_crediting_baseline.txt"

# --------------------------------------------------------------------------
# Project settings
# --------------------------------------------------------------------------
# Monitoring period (tx) in years — forward projection period,
# NOT the historical observation period. Must match the project's crediting period.
YEARS = 5


# =============================================================================
# Buffer: Create 10km donut buffer around project zone
# =============================================================================

def run_create_buffer(project_extent_gdf):
    """
    Create a 10km donut buffer around the project zone and save as KML.

    Args:
        project_extent_gdf: GeoDataFrame of the project extent
    """
    print("=" * 60)
    print("Buffer: Create 10km Donut Buffer")
    print("=" * 60)

    if project_extent_gdf.crs.is_projected:
        print(f"Project zone CRS is projected: {project_extent_gdf.crs}")
        project_extent_projected = project_extent_gdf.copy()
    else:
        print(f"Project zone CRS is geographic: {project_extent_gdf.crs}")
        centroid = project_extent_gdf.dissolve().centroid.iloc[0]
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = 'north' if centroid.y >= 0 else 'south'
        epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
        print(f"Reprojecting to UTM zone {utm_zone} ({hemisphere}): EPSG:{epsg_code}")
        project_extent_projected = project_extent_gdf.to_crs(epsg=epsg_code)

    project_extent_dissolved = project_extent_projected.dissolve()

    outer_buffer = project_extent_dissolved.copy()
    outer_buffer['geometry'] = project_extent_dissolved.geometry.buffer(10000)

    donut_gdf = outer_buffer.copy()
    donut_gdf['geometry'] = outer_buffer.geometry.difference(project_extent_dissolved.geometry)
    donut_gdf = donut_gdf.dissolve()
    donut_gdf = donut_gdf.to_crs(epsg=4326)

    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    donut_gdf.to_file(BUFFER_OUTPUT_PATH, driver='KML')

    print(f"10km buffer (donut, excluding project zone) saved to: {BUFFER_OUTPUT_PATH}")
    print("Buffer creation completed.\n")


# =============================================================================
# Load inputs
# =============================================================================

def run_load_inputs():
    """
    Load all input rasters and shapefiles needed for the pipeline.

    Returns:
        dict with keys: indicator_at_t0, original_nan_mask, p_x,
                        project_extent_gdf, prob_crs, prob_transform, height, width
    """
    print("=" * 60)
    print("Loading Inputs")
    print("=" * 60)

    project_extent_gdf = gpd.read_file(PROJECT_EXTENT)
    if project_extent_gdf.crs is None:
        raise ValueError("Project zone shapefile has no CRS defined.")

    with rasterio.open(RASTER_PATH) as src:
        raster_data = src.read()
        indicator_at_t0 = raster_data[-1, :, :]

    original_nan_mask = np.isnan(indicator_at_t0)
    indicator_at_t0[original_nan_mask] = 0

    with rasterio.open(PROBABILITY_MAP) as src:
        p_x = src.read(1)
        prob_crs = src.crs
        prob_transform = src.transform
        height = src.height
        width = src.width

    print(f"Indicator shape: {indicator_at_t0.shape}")
    print(f"Probability map CRS: {prob_crs}")
    print(f"Original NaN pixels: {original_nan_mask.sum()}")
    print("Inputs loaded.\n")

    return {
        'indicator_at_t0': indicator_at_t0,
        'original_nan_mask': original_nan_mask,
        'p_x': p_x,
        'project_extent_gdf': project_extent_gdf,
        'prob_crs': prob_crs,
        'prob_transform': prob_transform,
        'height': height,
        'width': width,
    }


# =============================================================================
# Step H4: Threshold probability map
# =============================================================================

def run_step_h4(p_x, prob_crs, prob_transform, height, width):
    """
    Step H4: Create high-risk probability raster (threshold >= 0.3).

    Args:
        p_x: Probability array (modified in-place)
        prob_crs: CRS of the probability map
        prob_transform: Transform of the probability map
        height: Raster height
        width: Raster width

    Returns:
        p_x: Modified probability array (thresholded)
    """
    print("=" * 60)
    print("Step H4: Threshold Probability Map")
    print("=" * 60)

    high_risk_p_x = p_x.copy()
    nan_mask = np.isnan(p_x)
    low_risk_mask = (p_x < 0.3) & ~nan_mask

    high_risk_p_x[nan_mask] = -1
    high_risk_p_x[low_risk_mask] = np.nan

    target_crs = "EPSG:4326"

    prob_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': prob_crs,
        'transform': prob_transform,
        'nodata': np.nan,
    }

    if prob_crs != target_crs:
        print(f"Reprojecting high-risk probability from {prob_crs} to {target_crs}")
        dst_transform, dst_width, dst_height = calculate_default_transform(
            prob_crs, target_crs, width, height,
            left=prob_transform.c,
            bottom=prob_transform.f + prob_transform.e * height,
            right=prob_transform.c + prob_transform.a * width,
            top=prob_transform.f
        )
        prob_profile.update(
            crs=target_crs, transform=dst_transform,
            width=dst_width, height=dst_height
        )
        high_risk_reprojected = np.empty((dst_height, dst_width), dtype='float32')
        reproject(
            source=high_risk_p_x.astype('float32'),
            destination=high_risk_reprojected,
            src_transform=prob_transform, src_crs=prob_crs,
            dst_transform=dst_transform, dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan
        )
        with rasterio.open(HIGH_RISK_PROB_PATH, 'w', **prob_profile) as dst:
            dst.write(high_risk_reprojected, 1)
    else:
        print(f"No reprojection needed - CRS already matches: {prob_crs}")
        prob_profile.update(crs=target_crs)
        with rasterio.open(HIGH_RISK_PROB_PATH, 'w', **prob_profile) as dst:
            dst.write(high_risk_p_x.astype('float32'), 1)

    print(f"High-risk probability raster saved to: {HIGH_RISK_PROB_PATH}")

    p_x[p_x < 0.3] = 0
    p_x[np.isnan(p_x)] = 0

    print("Step H4 completed.\n")
    return p_x


# =============================================================================
# Step H5: Remove exclusion zones
# =============================================================================

def run_step_h5(p_x, project_extent_gdf, prob_crs, transform, height, width):
    """
    Step H5: Remove exclusion zones from probability map.

    Args:
        p_x: Probability array (modified in-place)
        project_extent_gdf: GeoDataFrame of the project extent
        prob_crs: CRS of the probability map
        transform: Raster transform
        height: Raster height
        width: Raster width

    Returns:
        p_x: Modified probability array (exclusions applied)
    """
    print("=" * 60)
    print("Step H5: Remove Exclusion Zones")
    print("=" * 60)

    exclusion_mask = np.zeros_like(p_x, dtype=bool)

    if project_extent_gdf.crs.is_projected:
        project_extent_projected_h5 = project_extent_gdf.copy()
    else:
        centroid = project_extent_gdf.dissolve().centroid.iloc[0]
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = 'north' if centroid.y >= 0 else 'south'
        epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone
        project_extent_projected_h5 = project_extent_gdf.to_crs(epsg=epsg_code)

    project_extent_dissolved_h5 = project_extent_projected_h5.dissolve()

    # Exclusion 1: 10km donut
    outer_buffer_10km = project_extent_dissolved_h5.copy()
    outer_buffer_10km['geometry'] = project_extent_dissolved_h5.geometry.buffer(10000)
    donut_10km = outer_buffer_10km.copy()
    donut_10km['geometry'] = outer_buffer_10km.geometry.difference(project_extent_dissolved_h5.geometry)
    donut_10km = donut_10km.dissolve()

    # Exclusion 2: 500km outer limit
    buffer_500km = project_extent_dissolved_h5.copy()
    buffer_500km['geometry'] = project_extent_dissolved_h5.geometry.buffer(500000)
    buffer_500km = buffer_500km.dissolve()

    donut_10km = donut_10km.to_crs(prob_crs)
    buffer_500km = buffer_500km.to_crs(prob_crs)

    for geom in donut_10km.geometry:
        mask = features.geometry_mask(
            [geom], out_shape=(height, width), transform=transform, invert=True
        )
        exclusion_mask = exclusion_mask | mask
    print("10km donut exclusion applied (project zone NOT excluded)")

    outside_500km_mask = np.ones_like(p_x, dtype=bool)
    for geom in buffer_500km.geometry:
        mask = features.geometry_mask(
            [geom], out_shape=(height, width), transform=transform, invert=True
        )
        outside_500km_mask = outside_500km_mask & ~mask
    exclusion_mask = exclusion_mask | outside_500km_mask
    print("500km outer limit exclusion applied")

    # Exclusion 3: Exclusion zones from file
    if EXCLUSION_ZONES:
        exclusion_zones_gdf = gpd.read_file(EXCLUSION_ZONES)
        if exclusion_zones_gdf.crs != prob_crs:
            exclusion_zones_gdf = exclusion_zones_gdf.to_crs(prob_crs)
        for geom in exclusion_zones_gdf.geometry:
            mask = features.geometry_mask(
                [geom], out_shape=(height, width), transform=transform, invert=True
            )
            exclusion_mask = exclusion_mask | mask
        print("Exclusion zones from file applied")
    else:
        print("No exclusion zones file provided")

    with rasterio.open(PROBABILITY_MAP) as src:
        exclusion_profile = src.profile.copy()
    exclusion_profile.update(dtype='uint8', count=1, nodata=None)
    with rasterio.open(EXCLUSION_MASK_PATH, 'w', **exclusion_profile) as dst:
        dst.write(exclusion_mask.astype('uint8'), 1)
    print(f"Exclusion mask saved to: {EXCLUSION_MASK_PATH}")

    p_x[exclusion_mask] = 0
    print(f"Total excluded pixels: {exclusion_mask.sum()} / {exclusion_mask.size}")
    print("Step H5 completed.\n")

    return p_x


# =============================================================================
# Step H6: Proportion of high-risk pixels (binary count — SD VISta compliant)
# =============================================================================

def run_step_h6(p_x, project_extent_gdf, prob_crs, transform, height, width):
    """
    Step H6: Calculate proportion of high-risk pixels using binary count.

    Per SD VISta specification: HighRisk_p is a binary variable (0/1).
    w = count(high-risk pixels) / total project pixels.

    Args:
        p_x: Thresholded probability array
        project_extent_gdf: GeoDataFrame of the project extent
        prob_crs: CRS of the probability map
        transform: Raster transform
        height: Raster height
        width: Raster width

    Returns:
        Tuple of (w, project_mask)
    """
    print("=" * 60)
    print("Step H6: Proportion of High-Risk Pixels (Binary)")
    print("=" * 60)

    project_extent_for_mask = project_extent_gdf.to_crs(prob_crs)

    project_mask = np.zeros_like(p_x, dtype=bool)
    for geom in project_extent_for_mask.geometry:
        mask = features.geometry_mask(
            [geom], out_shape=(height, width), transform=transform, invert=True
        )
        project_mask = project_mask | mask

    print(f"Project mask pixels: {project_mask.sum()} / {project_mask.size}")

    # Binary count per SD VISta Step H6
    if project_mask.sum() == 0:
        print("WARNING: No pixels found in project zone mask. Check CRS alignment.")
        w = 0
    else:
        w = (p_x[project_mask] > 0).sum() / project_mask.sum()

    overlap_proportion = round(w, 3)

    with open(OVERLAP_PROPORTION_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Step H6: Proportion of high-risk pixels (binary count)\n")
        f.write(f"======================================================\n\n")
        f.write(f"High-risk pixels in project extent: {(p_x[project_mask] > 0).sum()}\n")
        f.write(f"Total pixels in project extent: {project_mask.sum()}\n")
        f.write(f"w (proportion): {overlap_proportion}\n")

    print(f"High-risk proportion (w): {overlap_proportion}")
    print(f"High-risk proportion saved to: {OVERLAP_PROPORTION_PATH}")
    print("Step H6 completed.\n")

    return w, project_mask


# =============================================================================
# Step H7: Estimated ecosystem condition (SD VISta compliant)
# =============================================================================

def run_step_h7(indicator_at_t0, p_x, project_mask, original_nan_mask):
    """
    Step H7: Calculate estimated ecosystem condition using SD VISta formula.

    I_hat(p,tx) = I(p,t0) x (1 - HighRiskProb_p)

    Models binary habitat conversion: with probability p the pixel is fully
    converted (value -> 0), with probability (1-p) it remains unchanged.

    Args:
        indicator_at_t0: Current indicator values
        p_x: Thresholded probability array
        project_mask: Boolean mask of project extent
        original_nan_mask: Boolean mask of original NaN pixels

    Returns:
        Tuple of (I_hat, high_risk_project_mask, reference_value)
    """
    print("=" * 60)
    print("Step H7: Estimated Ecosystem Condition (SD VISta)")
    print("=" * 60)

    high_risk_project_mask = project_mask & (p_x > 0) & ~original_nan_mask
    I_hat = np.zeros_like(indicator_at_t0, dtype='float32')
    I_hat[high_risk_project_mask] = (
        indicator_at_t0[high_risk_project_mask] * (1 - p_x[high_risk_project_mask])
    )

    # Save raster
    with rasterio.open(PROBABILITY_MAP) as src:
        h7_profile = src.profile.copy()
    h7_profile.update(dtype='float32', count=1, nodata=0)
    with rasterio.open(ESTIMATED_CONDITION_PATH, 'w', **h7_profile) as dst:
        dst.write(I_hat.astype('float32'), 1)

    print(f"Estimated ecosystem condition (I_hat) saved to: {ESTIMATED_CONDITION_PATH}")
    print(f"I_hat stats (high-risk pixels) - Min: {I_hat[high_risk_project_mask].min():.3f}, "
          f"Max: {I_hat[high_risk_project_mask].max():.3f}, Mean: {I_hat[high_risk_project_mask].mean():.3f}")

    # Reference value
    valid_indicator = indicator_at_t0[project_mask & ~original_nan_mask]
    reference_value = np.median(valid_indicator[valid_indicator > 0])

    indicator_at_t0[~project_mask] = 0

    print("Step H7 completed.\n")
    return I_hat, high_risk_project_mask, reference_value


# =============================================================================
# Steps H8-H11: Delta, Average, Standardize, Annualize
# =============================================================================

def run_steps_h8_to_h11(indicator_at_t0, I_hat, high_risk_project_mask, w, reference_value):
    """
    Steps H8-H11: Calculate delta, average, standardize, and annualize.

    Args:
        indicator_at_t0: Current indicator values
        I_hat: Estimated ecosystem condition
        high_risk_project_mask: Boolean mask of high-risk project pixels
        w: Proportion of high-risk pixels
        reference_value: Median indicator value in project extent

    Returns:
        ex_ante_baseline: Annual crediting baseline value
    """
    print("=" * 60)
    print("Steps H8-H11: Delta, Average, Standardize, Annualize")
    print("=" * 60)

    # Step H8: Delta
    delta_indicator = np.zeros_like(indicator_at_t0, dtype='float32')
    delta_indicator[high_risk_project_mask] = (
        I_hat[high_risk_project_mask] - indicator_at_t0[high_risk_project_mask]
    )

    with rasterio.open(PROBABILITY_MAP) as src:
        delta_profile = src.profile.copy()
    delta_profile.update(dtype='float32', count=1, nodata=0)
    with rasterio.open(DELTA_HIGH_RISK_PATH, 'w', **delta_profile) as dst:
        dst.write(delta_indicator, 1)

    print(f"Delta (high-risk pixels) saved to: {DELTA_HIGH_RISK_PATH}")
    print(f"High-risk pixels in project zone: {high_risk_project_mask.sum()}")
    print(f"Delta stats (high-risk only) - Min: {delta_indicator[high_risk_project_mask].min():.3f}, "
          f"Max: {delta_indicator[high_risk_project_mask].max():.3f}, "
          f"Mean: {delta_indicator[high_risk_project_mask].mean():.3f}")

    # Step H9: Average delta
    average_delta = delta_indicator[high_risk_project_mask].mean()

    with open(AVERAGE_DELTA_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Step H9: Average predicted change in structure indicator (high-risk pixels)\n")
        f.write(f"=========================================================================\n\n")
        f.write(f"Average delta (high-risk pixels): {average_delta:.6f}\n")
        f.write(f"Total high-risk pixels in project extent: {high_risk_project_mask.sum()}\n")
        f.write(f"Sum of delta values: {delta_indicator[high_risk_project_mask].sum():.6f}\n")

    print(f"Average delta (high-risk pixels): {average_delta}")
    print(f"Average delta saved to: {AVERAGE_DELTA_PATH}")

    # Step H10: Standardize
    deltaPrediction_ex_ante = average_delta * w / reference_value

    with open(STANDARDIZED_CHANGE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Step H10: Standardized, area-weighted average change in structure indicator\n")
        f.write(f"==========================================================================\n\n")
        f.write(f"Formula: dC_pre = (dI_tx * w) / Rv\n\n")
        f.write(f"dI_tx (average delta, Step H9): {average_delta:.6f}\n")
        f.write(f"w (high-risk proportion, Step H6): {w:.6f}\n")
        f.write(f"Rv (reference value): {reference_value:.6f}\n\n")
        f.write(f"dC_pre (standardized change): {deltaPrediction_ex_ante:.6f}\n")

    print(f"Standardized change (dC_pre): {deltaPrediction_ex_ante}")
    print(f"Standardized change saved to: {STANDARDIZED_CHANGE_PATH}")

    # Step H11: Annualize
    ex_ante_baseline = deltaPrediction_ex_ante / YEARS

    with open(CREDITING_BASELINE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Step H11: Estimated Crediting Baseline\n")
        f.write(f"======================================\n\n")
        f.write(f"Formula: B_pre = dC_pre / tx\n\n")
        f.write(f"dC_pre (standardized change, Step H10): {deltaPrediction_ex_ante:.6f}\n")
        f.write(f"tx (monitoring period in years): {YEARS}\n\n")
        f.write(f"B_pre (annual estimated crediting baseline): {ex_ante_baseline:.6f}\n")

    print(f"Crediting baseline (B_pre): {ex_ante_baseline}")
    print(f"Crediting baseline saved to: {CREDITING_BASELINE_PATH}")
    print("Steps H8-H11 completed.\n")

    return ex_ante_baseline


# =============================================================================
# Main Workflow
# =============================================================================

def run_crediting_baseline():
    """
    Run the complete crediting baseline calculation workflow.

    Uses SD VISta compliant formulas for Steps H6 and H7:
    - H6: binary count of high-risk pixels
    - H7: I_hat = I(t0) x (1 - HighRiskProb_p)
    """
    print("\n" + "=" * 60)
    print("CREDITING BASELINE CALCULATION (SD VISta Compliant)")
    print("=" * 60 + "\n")

    # Load inputs
    inputs = run_load_inputs()

    # Buffer
    run_create_buffer(inputs['project_extent_gdf'])

    # Step H4
    p_x = run_step_h4(
        inputs['p_x'], inputs['prob_crs'], inputs['prob_transform'],
        inputs['height'], inputs['width']
    )

    # Get transform for downstream steps
    with rasterio.open(PROBABILITY_MAP) as src:
        transform = src.transform

    # Step H5
    p_x = run_step_h5(
        p_x, inputs['project_extent_gdf'], inputs['prob_crs'],
        transform, inputs['height'], inputs['width']
    )

    # Step H6 (binary count)
    w, project_mask = run_step_h6(
        p_x, inputs['project_extent_gdf'], inputs['prob_crs'],
        transform, inputs['height'], inputs['width']
    )

    # Step H7 (SD VISta formula — no forecast_values needed)
    I_hat, high_risk_project_mask, reference_value = run_step_h7(
        inputs['indicator_at_t0'], p_x,
        project_mask, inputs['original_nan_mask']
    )

    # Steps H8-H11
    ex_ante_baseline = run_steps_h8_to_h11(
        inputs['indicator_at_t0'], I_hat, high_risk_project_mask,
        w, reference_value
    )

    print("\n" + "=" * 60)
    print("CREDITING BASELINE CALCULATION COMPLETED")
    print(f"Annual crediting baseline (B_pre): {ex_ante_baseline:.6f}")
    print("=" * 60 + "\n")

    return ex_ante_baseline


if __name__ == "__main__":
    # Run the full workflow
    run_crediting_baseline()

    # Or run individual steps (requires manual data passing):
    # inputs = run_load_inputs()
    # run_create_buffer(inputs['project_extent_gdf'])
    # p_x = run_step_h4(inputs['p_x'], ...)
    # ...
