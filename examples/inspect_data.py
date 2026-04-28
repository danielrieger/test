import pandas as pd
import os

data_path = 'examples/ShareLoc_Data/data.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print(f"Columns: {df.columns.tolist()}")
    print(f"X range: {df['x [nm]'].min()} to {df['x [nm]'].max()}")
    print(f"Y range: {df['y [nm]'].min()} to {df['y [nm]'].max()}")
    print(f"Total points: {len(df)}")
else:
    print(f"Data not found at {data_path}")
