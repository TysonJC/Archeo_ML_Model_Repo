import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.utils import resample

# # Example: Generate synthetic spatial data
# np.random.seed(42)
# n_points = 50
# x = np.random.uniform(0, 100, n_points)
# y = np.random.uniform(0, 100, n_points)
# z_raw = np.sin(x / 10) + np.cos(y / 10) + np.random.normal(0, 0.1, n_points)
# z = (z_raw - np.min(z_raw)) / (np.max(z_raw) - np.min(z_raw))

# # Define grid for interpolation
# grid_x = np.linspace(0, 100, 50)
# grid_y = np.linspace(0, 100, 50)

# mask = np.zeros((len(grid_y), len(grid_x)))

def run_ebk(x, y, z, grid_size=30, n_bootstrap=30, min_points=10, random_state=42):

# Bootstrapping variogram parameters (EBK-like)
n_bootstrap = 30
predictions = []

for _ in range(n_bootstrap):
    # Resample data with replacement
    x_res, y_res, z_res = resample(x, y, z)
    
    # Fit kriging model with random variogram parameters
    variogram_model = np.random.choice(['spherical', 'exponential'])

    coords = np.column_stack((x_res, y_res))
    _, unique_idx = np.unique(coords, axis=0, return_index=True)

    x_res = x_res[unique_idx]
    y_res = y_res[unique_idx]
    z_res = z_res[unique_idx]

    if len(x_res) < 15:
        continue
    
    try:
        OK = OrdinaryKriging(
            x_res, y_res, z_res,
            variogram_model=variogram_model,
            variogram_parameters={'nugget': 0.01, 'sill': 0.5, 'range': 50.0},
            verbose=False,
            enable_plotting=False
        )
        z_pred, _ = OK.execute('grid', grid_x, grid_y)
        predictions.append(z_pred)

        # Reject bad predictions BEFORE storing
        if z_pred is None or np.isnan(z_pred).all():
            continue

        if np.any(np.abs(z_pred) > 1e6):
            continue

    except Exception as e:
        print(f"Skipped iteration due to error: {e}")

# Mark original data points on grid
for xi, yi in zip(x, y):
    # Find closest grid index
    ix = np.argmin(np.abs(grid_x - xi))
    iy = np.argmin(np.abs(grid_y - yi))
    
    mask[iy, ix] = 1

# Combine predictions (mean of bootstraps)
predictions = np.array(predictions)
z_mean = np.nanmean(predictions, axis=0)
z_mean = np.nan_to_num(z_mean, nan=0.0, posinf=0.0, neginf=0.0)
z_std = np.nanstd(predictions, axis=0)  # Uncertainty estimate


# Save results
pd.DataFrame(z_mean).to_csv("ebk_mean_prediction.csv", index=False)
pd.DataFrame(z_std).to_csv("ebk_prediction_uncertainty.csv", index=False)
pd.DataFrame(mask).to_csv("ebk_original_points_mask.csv", index=False)
print("EBK-like interpolation complete. Results saved.")
