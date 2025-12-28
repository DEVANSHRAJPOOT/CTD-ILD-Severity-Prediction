import numpy as np
import nibabel as nib
import os

np.random.seed(42)

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
LABEL_DIR = os.path.join(BASE_DIR, "data", "labels")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

NUM_PATIENTS = 10
SHAPE = (128, 128, 128)

for i in range(NUM_PATIENTS):
    # Simulated CT image (HU-like values)
    ct = np.random.normal(loc=-700, scale=300, size=SHAPE)

    # Segmentation mask
    seg = np.zeros(SHAPE, dtype=np.uint8)

    # Random lesion blobs
    for label, prob in [(2, 0.01), (3, 0.008), (4, 0.004)]:
        mask = np.random.rand(*SHAPE) < prob
        seg[mask] = label

    affine = np.eye(4)

    img_nii = nib.Nifti1Image(ct.astype(np.float32), affine)
    seg_nii = nib.Nifti1Image(seg.astype(np.uint8), affine)

    nib.save(img_nii, f"{IMAGE_DIR}/patient_{i:03d}.nii.gz")
    nib.save(seg_nii, f"{LABEL_DIR}/patient_{i:03d}_seg.nii.gz")

print(f"Generated {NUM_PATIENTS} synthetic CT volumes and labels.")
