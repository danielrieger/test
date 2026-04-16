import pandas as pd
import numpy as np
import mrcfile
import json
import os
from scipy.ndimage import gaussian_filter

def render_smlm_to_mrc(csv_path, mrc_output_path, json_output_path, pixel_size_nm=10, sigma_px=2.0):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    x = df['x [nm]'].values
    y = df['y [nm]'].values

    # Determine bounds
    min_x, max_x = 0, np.ceil(x.max() / pixel_size_nm) * pixel_size_nm
    min_y, max_y = 0, np.ceil(y.max() / pixel_size_nm) * pixel_size_nm

    width_px = int(max_x / pixel_size_nm) + 1
    height_px = int(max_y / pixel_size_nm) + 1

    print(f"Rendering {width_px}x{height_px} image ({pixel_size_nm} nm/pixel)...")

    # Create 2D histogram
    # Note: np.histogram2d uses (x, y) order, which corresponds to (column, row)
    # We want the output to be (height, width)
    heatmap, xedges, yedges = np.histogram2d(
        y, x, 
        bins=[height_px, width_px], 
        range=[[0, height_px * pixel_size_nm], [0, width_px * pixel_size_nm]]
    )

    # Apply Gaussian blur to make it look like a micrograph
    print(f"Applying Gaussian blur (sigma={sigma_px} px)...")
    heatmap = gaussian_filter(heatmap, sigma=sigma_px)

    # Normalize and Invert for EMAN2 (dark particles on light background)
    # Scale to 0-1 first
    h_max = heatmap.max()
    if h_max > 0:
        heatmap /= h_max
    
    # Invert: EMAN2 usually expects particles to be darker than background in many picking modes
    # or at least a standard contrast. 1.0 - heatmap makes the dense areas (NPCs) dark.
    heatmap = 1.0 - heatmap

    # Save as MRC
    print(f"Saving MRC to {mrc_output_path}...")
    with mrcfile.new(mrc_output_path, overwrite=True) as mrc:
        mrc.set_data(heatmap.astype(np.float32))
        mrc.voxel_size = pixel_size_nm * 10.0 # EMAN2 often uses Angstroms in header

    # Save pixel map for reverse translation
    mapping = {
        "pixel_size_nm": pixel_size_nm,
        "width_px": width_px,
        "height_px": height_px,
        "min_x": 0,
        "min_y": 0,
        "max_x": float(max_x),
        "max_y": float(max_y)
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
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    render_smlm_to_mrc(args.input, args.output_mrc, args.output_json, args.pixel_size, args.sigma)
