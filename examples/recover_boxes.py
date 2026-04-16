import os
import json
import pandas as pd
import numpy as np
import glob

def recover_eman2_boxes(picked_dir="picked_particles", pixel_map_path="pixel_map.json", output_json="info/micrograph_info.json"):
    if not os.path.exists(picked_dir):
        print(f"Error: Could not find {picked_dir}")
        return
        
    with open(pixel_map_path, 'r') as f:
        pixel_map = json.load(f)
    
    pixel_size_nm = pixel_map['pixel_size_nm']
    global_boxsize = 32
    
    csv_files = glob.glob(os.path.join(picked_dir, "particle_*.csv"))
    if len(csv_files) == 0:
        print("No particle CSV files found to recover from.")
        return
        
    print(f"Found {len(csv_files)} particles. Recovering EMAN2 box centers...")
    
    boxes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
            
        # Calculate the geometric center of the particle in SMLM space
        # This will securely anchor the EMAN2 box precisely on the target
        mean_x_nm = df['x [nm]'].mean()
        mean_y_nm = df['y [nm]'].mean()
        
        # Convert nm back to EMAN2 pixel coordinates
        px_x = mean_x_nm / pixel_size_nm
        px_y = mean_y_nm / pixel_size_nm
        
        # EMAN2 format [x, y, type, confidence]
        boxes.append([float(px_x), float(px_y), "manual", 1.0])
        
    # Sort boxes by X then Y just to be deterministic
    boxes.sort(key=lambda b: (b[0], b[1]))
    
    # Structure the EMAN2 JSON
    info_data = {
        "global.boxsize": global_boxsize,
        "boxes": boxes
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(info_data, f, indent=4)
        
    print(f"Successfully recovered {len(boxes)} coordinate frames.")
    print(f"Saved exact EMAN2 replica to {output_json}")

if __name__ == "__main__":
    recover_eman2_boxes()
