import pandas as pd
import numpy as np
import mrcfile
import json
import os
from scipy.ndimage import gaussian_filter

def render_smlm_to_mrc(csv_path, mrc_output_path, json_output_path, pixel_size_nm=5.0, sigma_px=1.5, use_intensity=True):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    x = df['x [nm]'].values
    y = df['y [nm]'].values
    
    # Use Amplitude_0_0 as weights if available
    weights = None
    if use_intensity and 'Amplitude_0_0' in df.columns:
        print("Using 'Amplitude_0_0' as intensity weights.")
        weights = df['Amplitude_0_0'].values
    else:
        print("Using localization counts as intensity.")

    # Determine bounds
    min_x, max_x = 0, np.ceil(x.max() / pixel_size_nm) * pixel_size_nm
    min_y, max_y = 0, np.ceil(y.max() / pixel_size_nm) * pixel_size_nm

    width_px = int(max_x / pixel_size_nm) + 1
    height_px = int(max_y / pixel_size_nm) + 1

    print(f"Rendering {width_px}x{height_px} image ({pixel_size_nm} nm/pixel)...")

    # Create 2D histogram
    # To avoid vertical flip in EMAN2, we render such that y=0 is at the BOTTOM.
    # np.histogram2d(x, y) returns H[i, j] for x[i], y[j].
    # We want a (height, width) matrix where rows correspond to Y.
    heatmap, xedges, yedges = np.histogram2d(
        y, x, 
        bins=[height_px, width_px], 
        range=[[0, height_px * pixel_size_nm], [0, width_px * pixel_size_nm]],
        weights=weights
    )

    # Apply Gaussian blur
    print(f"Applying Gaussian blur (sigma={sigma_px} px)...")
    heatmap = gaussian_filter(heatmap, sigma=sigma_px)

    # Normalize
    h_max = heatmap.max()
    if h_max > 0:
        heatmap /= h_max
    
    # IMPORTANT: EMAN2 / MRC viewers usually treat the first element of the array as the BOTTOM-LEFT.
    # NumPy's default display of a matrix puts [0,0] at the TOP-LEFT.
    # Our histogram already has y=0 at index 0, which is the BOTTOM in MRC terms.
    # So we do NOT need to flip it if we want y=0 at the bottom.
    
    # Invert for EMAN2 (dark particles on light background)
    heatmap = 1.0 - heatmap

    # Save as MRC
    print(f"Saving MRC to {mrc_output_path}...")
    with mrcfile.new(mrc_output_path, overwrite=True) as mrc:
        # MRC data is typically (Z, Y, X). For 2D, (Y, X).
        mrc.set_data(heatmap.astype(np.float32))
        # Set pixel size in Angstroms
        mrc.voxel_size = pixel_size_nm * 10.0

    # Save pixel map for reverse translation
    mapping = {
        "pixel_size_nm": pixel_size_nm,
        "width_px": width_px,
        "height_px": height_px,
        "min_x": 0,
        "min_y": 0,
        "max_x": float(max_x),
        "max_y": float(max_y),
        "y_is_flipped": False # Explicitly tracking this for the extraction script
    }
    print(f"Saving pixel map to {json_output_path}...")
    with open(json_output_path, 'w') as f:
        json.dump(mapping, f, indent=4)

    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="examples/ShareLoc_Data/data.csv")
    parser.add_argument("--output_mrc", default="examples/micrograph.mrc")
    parser.add_argument("--output_json", default="examples/pixel_map.json")
    parser.add_argument("--pixel_size", type=float, default=5.0) # Higher res
    parser.add_argument("--sigma", type=float, default=1.5)
    args = parser.parse_args()

    render_smlm_to_mrc(args.input, args.output_mrc, args.output_json, args.pixel_size, args.sigma)
