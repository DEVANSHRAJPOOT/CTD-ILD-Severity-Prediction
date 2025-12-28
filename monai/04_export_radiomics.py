import os
import nibabel as nib
import numpy as np
import pandas as pd

# Base project directory (absolute path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASK_DIR = os.path.join(BASE_DIR, "data", "predicted_masks")
OUT_CSV = os.path.join(BASE_DIR, "data", "radiology.csv")

rows = []

if not os.path.exists(MASK_DIR):
    raise RuntimeError(f"Predicted mask directory not found: {MASK_DIR}")

for fname in os.listdir(MASK_DIR):
    if not fname.endswith(".nii.gz"):
        continue

    path = os.path.join(MASK_DIR, fname)
    seg = nib.load(path)
    data = seg.get_fdata()
    voxel_volume = np.prod(seg.header.get_zooms())

    row = {
        "PatientID": fname.replace("_seg.nii.gz", ""),
        "GGO_Volume": np.sum(data == 2) * voxel_volume,
        "Fibrosis_Volume": np.sum(data == 3) * voxel_volume,
        "Honeycombing": np.sum(data == 4) * voxel_volume,
        "Segments_Involved": int(np.sum(np.unique(data) > 0))
    }

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print(f"Radiology features saved to: {OUT_CSV}")
print(df.head())
