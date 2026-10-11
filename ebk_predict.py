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

    #Convert inputs to numpy arrays

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    #Ensure each array has the same length
    if (len(x) != len(y) or len(y) != len(z)):
        raise ValueError("Input arrays x, y, and z must have the same length.")

    #Ensure no invalid input
    valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)

    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]

    #Ensure the predictions are between 0 and 1
    z = np.clip(z, 0.0, 0.1)

    #Address conflicts with similar coordinates

    origin_points = pd.DataFrame({
        "x_coord": x,
        "y_coord": y,
        "non_soil_probability": z
    })

    #Multiple samples could be taken at the same area
    #Within similar coordinates with predictions, we'll take the mean value of those predictions
    
    unique_mean_points = (original_points_df.groupby(
            ["x_coord", "y_coord"],
            as_index=False
    )
        ["non_soil_probability"]
        .mean()
    )

    x = unique_mean_points["x_coord"].to_numpy()
    y = unique_mean_points["y_coord"].to_numpy()
    z = unique_mean_points["non_soil_probability"].to_numpy()

    #Establish the grid

    x_grid = np.linspace(np.min(x),np.max(x),grid_size)
    y_grid = np.linspace(np.min(y),np.max(y),grid_size)

    rng = np.random.default_rng(random_state)

    predictions = []

    variogram_model = ['spherical', 'exponential', 'gaussian']
    
    for num_runs in range(n_bootstrap):
        # Get the sample of the original point
        origin_p = rng.integers(0,len(x),size=len(x))
        
        x_res = x_res[origin_p]
        y_res = y_res[origin_p]
        z_res = z_res[origin_p]

        #Remove duplicates from the bootstrap

        grid_resampled = pd.DataFrame({
            'x': x_res,
            'y': y_res,
            'z': z_res
        })

        resampled_points = (grid_resampled.groupby(
            ["x", "y"],
            as_index=False
        )
        ["z"]
        .mean()
        )

        x_res = resampled_points["x"].to_numpy()
        y_res = resampled_points["y"].to_numpy()
        z_res = resampled_points["z"].to_numpy()

        if len(x_res) < min_points:
            continue

        #Select different Variogram models each run
        variogram_model = str(
            rng.choice(variogram_models))
        
        try:
            kriging_model = OrdinaryKriging(
                x_res,
                y_res,
                z_res,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )

            z_pred, variance = kriging_model.execute(
                "grid",
                grid_x,
                grid_y
            )

            z_pred = np.asarray(
                np.ma.filled(z_pred, np.nan),
                dtype=float
            )

            # Reject bad predictions BEFORE storing
            if z_pred is None or np.isnan(z_pred).all():
                continue
    
            if np.any(np.abs(z_pred) > 1e6):
                continue

            # Probability cannot logically exceed 0-1
            z_pred = np.clip(
                z_pred,
                0.0,
                1.0
            )

            predictions.append(z_pred)
    
        except Exception as e:
            print(f"Skipped iteration due to error: {e}")


    #Check if one model appears
    
    if len(predictions) == 0:
        raise RuntimeError("All EBK interpolation attempts failed.")


    # Combine predictions (mean of bootstraps)
    predictions = np.array(predictions)
    z_mean = np.nanmean(predictions, axis=0)
    z_mean = np.nan_to_num(z_mean, nan=0.0, posinf=0.0, neginf=0.0)
    z_mean = np.clip(mean_grid,0.0,1.0)
    z_uncertain = np.nanstd(predictions, axis=0)  # Uncertainty estimate
    
    
    # # Mark original data points on grid
    # for xi, yi in zip(x, y):
    #     # Find closest grid index
    #     ix = np.argmin(np.abs(grid_x - xi))
    #     iy = np.argmin(np.abs(grid_y - yi))
        
    #     mask[iy, ix] = 1

    #TODO: Convert to a set grid (30 x 30)

    x_grid_mesh, y_grid_mesh = np.meshgrid(x_grid,y_grid)

    grid_display = pd.DataFrame({
        "x_coord": grid_x_mesh.flatten(),
        "y_coord": grid_y_mesh.flatten(),
        "ebk_probability": mean_grid.flatten(),
        "ebk_uncertainty": uncertainty_grid.flatten()
    })

    print("EBK completed")
    
    return grid_display, origin_points

    #TODO: Helper Function to be used in main.py
    #Will take the dataframe to produce the grid

def dataframe_to_grid(dataframe, x_col, y_col, prob_col):
    
    req_columns = [x_column,y_column,value_column]

    #Check if any columns are missing

    missing_columns = [
        column
        for column in req_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "EBK requires these missing columns: "+ ", ".join(missing_columns))

    return run_ebk(
        x=dataframef[x_col],
        y=dataframe[y_col],
        z=dataframe[prob_column],
        grid_size=30,
        n_bootstrap=30
    )

    
# # Save results
#     pd.DataFrame(z_mean).to_csv("ebk_mean_prediction.csv", index=False)
#     pd.DataFrame(z_std).to_csv("ebk_prediction_uncertainty.csv", index=False)
#     pd.DataFrame(mask).to_csv("ebk_original_points_mask.csv", index=False)
#     print("EBK-like interpolation complete. Results saved.")
    
