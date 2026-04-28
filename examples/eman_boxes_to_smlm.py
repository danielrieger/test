import pandas as pd
import numpy as np
import json
import os

def extract_smlm_from_boxes(csv_path, boxes_json_path, pixel_map_path, output_dir="picked_particles"):
    if not os.path.exists(boxes_json_path):
        print(f"Error: Box file {boxes_json_path} not found. Did you run EMAN2 picking?")
        return

    print(f"Loading boxes from {boxes_json_path}...")
    with open(boxes_json_path, 'r') as f:
        info_data = json.load(f)
    
    # EMAN2 stores boxes in a list, often under the 'boxes' or 'boxes_3d' key
    # Sometimes it's a flat list in info files.
    boxes = info_data.get('boxes', [])
    if not boxes:
        # Fallback: check if the json IS the list of boxes
        if isinstance(info_data, list):
            boxes = info_data
        else:
            print("Warning: No boxes found in JSON. Checking other common EMAN2 keys...")
            # Look for keys containing 'boxes'
            for k in info_data.keys():
                if 'boxes' in k.lower():
                    boxes = info_data[k]
                    print(f"Using key: {k}")
                    break

    # EMAN2 usually stores a global box size in the project.json or top level of info
    project_box_size = info_data.get('global.boxsize', 32)
    
    if not boxes:
        print("Error: No boxes found in JSON file.")
        return

    print(f"Loading pixel map from {pixel_map_path}...")
    with open(pixel_map_path, 'r') as f:
        pixel_map = json.load(f)
    
    pixel_size_nm = pixel_map['pixel_size_nm']

    print(f"Loading original SMLM data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing {len(boxes)} boxes using box_size={project_box_size}...")
    for i, box in enumerate(boxes):
        # EMAN2 Auto-box format: [x_center, y_center, type, confidence_score]
        px_x = box[0]
        px_y = box[1]
        
        # Use project_box_size instead of box[3] (which is a score for auto-pick)
        box_size_px = project_box_size
        
        half_size_nm = (box_size_px / 2.0) * pixel_size_nm
        x_min = (px_x * pixel_size_nm) - half_size_nm
        x_max = (px_x * pixel_size_nm) + half_size_nm
        y_min = (px_y * pixel_size_nm) - half_size_nm
        y_max = (px_y * pixel_size_nm) + half_size_nm
        
        # Slice dataframe
        mask = (df['x [nm]'] >= x_min) & (df['x [nm]'] <= x_max) & \
               (df['y [nm]'] >= y_min) & (df['y [nm]'] <= y_max)
        
        cluster_df = df[mask]
        
        if not cluster_df.empty:
            out_file = os.path.join(output_dir, f"particle_{i}.csv")
            cluster_df.to_csv(out_file, index=False)
            # print(f"Saved {len(cluster_df)} points to {out_file}")
        else:
            pass # print(f"Box {i} is empty in SMLM space.")

    print(f"Extraction complete. Check the '{output_dir}' directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="examples/ShareLoc_Data/data.csv")
    parser.add_argument("--input_boxes", default="examples/info/micrograph_info.json")
    parser.add_argument("--pixel_map", default="examples/pixel_map.json")
    parser.add_argument("--output_dir", default="examples/picked_particles")
    args = parser.parse_args()

    extract_smlm_from_boxes(args.input_csv, args.input_boxes, args.pixel_map, args.output_dir)
