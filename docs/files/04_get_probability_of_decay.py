"""
Probability of Decay Calculation - SD VISta Nature Framework v1.0
==================================================================

This script generates spatially explicit probability maps and forecast values for
habitat conversion risk assessment.

METHODOLOGY:
-----------
1. Fits linear regression model for each pixel using:
   - Historical time series data (response variable)
   - Static drivers (e.g., human pressure index)
   - Optional dynamic drivers (e.g., climate variables)

2. Generates TWO outputs:
   a) FORECAST VALUES (forecast_values.tif):
      - Predicted indicator value at t+1 based on historical trends
      - Used to calculate magnitude of expected change
      - Based on extrapolating historical conversion rates

   b) PROBABILITY MAP (probability_of_decay.tif):
      - Probability that future value will be below current value
      - Used to identify high-risk pixels (threshold >= 0.3)
      - Used to weight the crediting baseline by risk exposure

COMPLIANCE:
----------
This approach is compliant with SD VISta Nature Framework v1.0 requirement to:
"model spatially explicit probabilities of habitat conversion using historical
conversion rates along with predictors of habitat loss"

The regression model embeds historical conversion rates through the time trend
coefficient, and probability is calculated from the forecast uncertainty.

IMPORTANT DISTINCTION:
---------------------
- PROBABILITY: Likelihood that decline will occur (0-1 scale)
- FORECAST: Amount of decline if it occurs (same units as indicator)

These are separate measures and should not be conflated.

For detailed methodology documentation, see:
METHODOLOGY_COMPLIANCE_AND_CHANGES.md

Last Updated: 2025-10-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import rasterio
from rasterio.windows import Window
import os
import time
from sklearn.linear_model import LinearRegression
import gc


# =============================================================================
# Configuration
# =============================================================================

# --------------------------------------------------------------------------
# Input paths
# --------------------------------------------------------------------------
RESPONSE_PATH = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\response\MOD44B_TreeCover_2014-2023.tif"
DYNAMIC_DRIVER_PATH = None  # r"A:\...\MODIS_NPP_10yr.tif"
STATIC_DRIVER_PATH = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\ancillary\CISI_NatFrame.tif"

# --------------------------------------------------------------------------
# Output folder
# --------------------------------------------------------------------------
OUTPUT_FOLDER = r"A:\TGC-01925 Mexico-ACAC-Yucatan-Biodiversity Credits\2_RS\Crediting_baseline\outputs"

# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------
OUTPUT_FORECAST_PATH = f"{OUTPUT_FOLDER}\\StepH3-2_forecast_values.tif"
OUTPUT_PROB_PATH = f"{OUTPUT_FOLDER}\\StepH3-1_probability_of_decay.tif"
OUTPUT_POINTS_XLSX = f"{OUTPUT_FOLDER}\\StepH3_model_points.xlsx"

# --------------------------------------------------------------------------
# Processing parameters
# --------------------------------------------------------------------------
MIN_TREE_COVER = 10  # Minimum tree cover percentage to be considered "forested"
CHUNK_SIZE = 1000    # Process in chunks of 1000x1000 pixels


# =============================================================================
# Step H3: Process chunk — per-pixel linear regression
# =============================================================================

def process_chunk(window, response_data, dynamic_data, static_data, n_timesteps, time_idx,
                   collect_points=False, transform=None):
    """
    Process a chunk of the raster data to generate forecasts and probability maps.

    Args:
        window: The rasterio window defining the chunk
        response_data: Time series data for response variable (n_timesteps, h, w)
        dynamic_data: Time series data for dynamic driver (n_timesteps, h, w) or None
        static_data: Static driver data (1, h, w)
        n_timesteps: Number of time steps
        time: Time array (0, 1, ..., n_timesteps-1)
        collect_points: If True, collect per-pixel model data for Excel export
        transform: Raster transform (required when collect_points=True)

    Returns:
        Tuple of (forecast_chunk, probability_chunk, points_list)
        points_list is empty if collect_points=False
    """
    chunk_height, chunk_width = window.height, window.width

    response_reshaped = response_data.reshape(n_timesteps, -1)
    if dynamic_data is not None:
        dynamic_reshaped = dynamic_data.reshape(n_timesteps, -1)
    static_reshaped = static_data.reshape(1, -1).repeat(n_timesteps, axis=0)

    n_pixels = chunk_height * chunk_width
    forecast_values = np.full(n_pixels, np.nan)
    prob_values = np.full(n_pixels, np.nan)
    points_list = []

    valid_mask = response_reshaped[-1, :] >= MIN_TREE_COVER
    report_every = max(1, n_pixels // 10)   # print every ~10 % of all pixels
    processed_valid = 0

    for pixel_idx in range(n_pixels):
        if pixel_idx > 0 and pixel_idx % report_every == 0:
            pct = 100 * pixel_idx / n_pixels
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}]  chunk pixels {pct:3.0f}%")
        if not valid_mask[pixel_idx]:
            continue

        y = response_reshaped[:, pixel_idx]

        if dynamic_data is not None:
            X_dynamic = dynamic_reshaped[:, pixel_idx].reshape(-1, 1)
            X_static = static_reshaped[:, pixel_idx].reshape(-1, 1)
            X = np.hstack([X_dynamic, X_static, time_idx.reshape(-1, 1)])
        else:
            X_static = static_reshaped[:, pixel_idx].reshape(-1, 1)
            X = np.hstack([X_static, time_idx.reshape(-1, 1)])

        if np.isnan(y).any() or np.isnan(X).any():
            continue

        model = LinearRegression()
        try:
            model.fit(X, y)

            if dynamic_data is not None:
                X_next = np.array([[
                    dynamic_reshaped[-1, pixel_idx],
                    static_reshaped[0, pixel_idx],
                    n_timesteps
                ]])
            else:
                X_next = np.array([[
                    static_reshaped[0, pixel_idx],
                    n_timesteps
                ]])

            forecast_value = model.predict(X_next)[0]

            y_pred = model.predict(X)
            residuals = y - y_pred
            dof = len(y) - X.shape[1] - 1
            if dof < 1:
                dof = 1
            mse = np.sum(residuals**2) / dof
            std_err = np.sqrt(mse)

            last_value = y[-1]

            if std_err < 1e-10:
                prob = 1.0 if forecast_value < last_value else 0.0
            else:
                prob = norm.cdf(last_value, loc=forecast_value, scale=std_err)

            forecast_values[pixel_idx] = forecast_value
            prob_values[pixel_idx] = prob
            processed_valid += 1

            if collect_points:
                # Convert pixel index to row/col within chunk, then to global row/col
                local_row = pixel_idx // chunk_width
                local_col = pixel_idx % chunk_width
                global_row = window.row_off + local_row
                global_col = window.col_off + local_col

                x_geo, y_geo = rasterio.transform.xy(transform, global_row, global_col, offset='center')

                point = {
                    'row': global_row,
                    'col': global_col,
                    'longitude': x_geo,
                    'latitude': y_geo,
                }

                for t in range(n_timesteps):
                    point[f'response_t{t}'] = float(y[t])

                point['static_driver'] = float(static_reshaped[0, pixel_idx])

                if dynamic_data is not None:
                    for t in range(n_timesteps):
                        point[f'dynamic_driver_t{t}'] = float(dynamic_reshaped[t, pixel_idx])

                point['forecast'] = float(forecast_value)
                point['probability'] = float(prob)
                point['std_error'] = float(std_err)

                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                point['r2'] = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else None

                point['intercept'] = float(model.intercept_)
                for c_idx, coef in enumerate(model.coef_):
                    point[f'coef_{c_idx}'] = float(coef)

                points_list.append(point)

        except Exception as e:
            print(f"Error processing pixel {pixel_idx}: {e}")

    forecast_chunk = forecast_values.reshape(chunk_height, chunk_width)
    prob_chunk = prob_values.reshape(chunk_height, chunk_width)

    return forecast_chunk, prob_chunk, points_list


# =============================================================================
# Step H3: Generate probability and forecast maps
# =============================================================================

def run_step_h3():
    """
    Step H3: Generate spatially explicit probability of decay and forecast maps.

    Reads the response time series, static/dynamic drivers, fits per-pixel
    linear regressions in chunks, and writes forecast + probability GeoTIFFs.
    Also exports per-pixel model data (coordinates, variables, outputs,
    diagnostics) to Excel. If valid points exceed Excel's row limit, data
    is split across multiple sheets.
    """
    print("=" * 60)
    print("Step H3: Probability of Decay and Forecast Maps")
    print("=" * 60)

    with rasterio.open(RESPONSE_PATH) as src:
        n_timesteps = src.count
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    time_idx = np.arange(n_timesteps)

    n_row_chunks = (height + CHUNK_SIZE - 1) // CHUNK_SIZE
    n_col_chunks = (width + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = n_row_chunks * n_col_chunks

    print(f"Response raster: {n_timesteps} timesteps, {height}x{width} pixels")
    print(f"Chunk size: {CHUNK_SIZE}x{CHUNK_SIZE}  |  Total chunks: {total_chunks}")
    print(f"Min tree cover threshold: {MIN_TREE_COVER}%")
    all_points = []
    chunk_idx = 0

    with rasterio.open(
        OUTPUT_FORECAST_PATH, 'w',
        driver='GTiff', height=height, width=width, count=1,
        dtype='float32', crs=crs, transform=transform, nodata=np.nan
    ) as dst_forecast, rasterio.open(
        OUTPUT_PROB_PATH, 'w',
        driver='GTiff', height=height, width=width, count=1,
        dtype='float32', crs=crs, transform=transform, nodata=np.nan
    ) as dst_prob:

        for row in range(0, height, CHUNK_SIZE):
            for col in range(0, width, CHUNK_SIZE):
                chunk_idx += 1
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}]  chunk {chunk_idx}/{total_chunks}  ({100*chunk_idx/total_chunks:3.0f}%)")

                window = Window(
                    col, row,
                    min(CHUNK_SIZE, width - col),
                    min(CHUNK_SIZE, height - row)
                )

                with rasterio.open(RESPONSE_PATH) as src_x:
                    response_data = src_x.read(window=window)
                if DYNAMIC_DRIVER_PATH is not None:
                    with rasterio.open(DYNAMIC_DRIVER_PATH) as src_b:
                        dynamic_data = src_b.read(window=window)
                else:
                    dynamic_data = None
                with rasterio.open(STATIC_DRIVER_PATH) as src_c:
                    static_data = src_c.read(window=window)

                forecast_chunk, prob_chunk, points_list = process_chunk(
                    window, response_data, dynamic_data, static_data,
                    n_timesteps, time_idx,
                    collect_points=True, transform=transform
                )

                dst_forecast.write(forecast_chunk.astype('float32'), 1, window=window)
                dst_prob.write(prob_chunk.astype('float32'), 1, window=window)
                all_points.extend(points_list)

                del response_data, dynamic_data, static_data, forecast_chunk, prob_chunk
                gc.collect()

    print(f"Forecast values saved to: {OUTPUT_FORECAST_PATH}")
    print(f"Probability map saved to: {OUTPUT_PROB_PATH}")

    _write_points_to_excel(all_points)

    print("Step H3 completed.\n")


# =============================================================================
# Export model points to Excel (helper for run_step_h3)
# =============================================================================

EXCEL_MAX_ROWS = 1_048_576  # Excel row limit per sheet (including header)

def _write_points_to_excel(all_points):
    """
    Write collected model points to Excel, splitting across sheets if needed.

    Args:
        all_points: List of dictionaries, one per valid pixel, with coordinates,
            variables, model outputs, and diagnostics.
    """
    total_points = len(all_points)
    print(f"\nValid points collected: {total_points}")

    if total_points == 0:
        print("WARNING: No valid points found. Excel not created.")
        return

    df = pd.DataFrame(all_points)

    max_data_rows = EXCEL_MAX_ROWS - 1  # Reserve 1 row for header
    n_sheets = 1 if total_points <= max_data_rows else (total_points + max_data_rows - 1) // max_data_rows

    print(f"Writing to Excel: {n_sheets} sheet(s)...")

    with pd.ExcelWriter(OUTPUT_POINTS_XLSX, engine='openpyxl') as writer:
        for sheet_idx in range(n_sheets):
            start = sheet_idx * max_data_rows
            end = min(start + max_data_rows, total_points)
            sheet_name = f"Points_{sheet_idx + 1}" if n_sheets > 1 else "Points"

            df.iloc[start:end].to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name}: rows {start + 1} to {end}")

    print(f"Excel saved to: {OUTPUT_POINTS_XLSX}")


# =============================================================================
# Visualization
# =============================================================================

def run_visualize_probability():
    """Display the probability of decay map."""
    print("=" * 60)
    print("Visualizing Probability Map")
    print("=" * 60)

    with rasterio.open(OUTPUT_PROB_PATH) as src:
        prob_map = src.read(1)

    plt.figure(figsize=(10, 8))
    plt.imshow(prob_map, cmap='viridis')
    plt.colorbar(label='Probability of Decrease')
    plt.title('Probability Map for Future Value Decrease')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()


# =============================================================================
# Main Workflow
# =============================================================================

def run_probability_workflow():
    """
    Run the complete probability of decay workflow:
    1. Generate forecast and probability maps (Step H3)
    2. Visualize the probability map
    """
    print("\n" + "=" * 60)
    print("PROBABILITY OF DECAY WORKFLOW")
    print("=" * 60 + "\n")

    run_step_h3()
    run_visualize_probability()

    print("\n" + "=" * 60)
    print("PROBABILITY OF DECAY WORKFLOW COMPLETED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run the full workflow
    run_probability_workflow()

    # Or run individual steps:
    # run_step_h3()
    # run_visualize_probability()
