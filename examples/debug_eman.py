import os
import sys

print(f"Current working directory: {os.getcwd()}")
path = "/home/daniel/Thesis/smlm_score/examples/micrograph.mrc"
print(f"Checking path: {path}")
print(f"Exists: {os.path.exists(path)}")
print(f"Readable: {os.access(path, os.R_OK)}")

try:
    import mrcfile
    with mrcfile.open(path) as mrc:
        print(f"mrcfile read success: {mrc.data.shape}")
except ImportError:
    print("mrcfile not installed in this env.")
except Exception as e:
    print(f"mrcfile error: {e}")

try:
    from EMAN2 import EMData
    print("EMAN2 imported successfully.")
    d = EMData(path)
    print(f"EMAN2 EMData read success: {d.get_x()}x{d.get_y()}")
except Exception as e:
    print(f"EMAN2 EMData error: {e}")
    # Try relative path
    try:
        rel_path = "micrograph.mrc"
        d = EMData(rel_path)
        print(f"EMAN2 EMData relative read success: {d.get_x()}x{d.get_y()}")
    except Exception as e2:
        print(f"EMAN2 EMData relative error: {e2}")
