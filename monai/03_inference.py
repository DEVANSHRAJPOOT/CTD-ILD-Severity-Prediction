import os
import sys
import torch
import nibabel as nib
import numpy as np

from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged
)
from monai.data import Dataset, DataLoader
from monai.transforms import AsDiscrete

# Fix import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
IMAGE_DIR = os.path.join("data", "images")
OUT_DIR = os.path.join("data", "predicted_masks")
MODEL_PATH = os.path.join("monai", "unet_ctd_ild.pth")

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transforms (NO cropping for inference)
infer_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5)),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=400,
        b_min=0.0, b_max=1.0,
        clip=True
    )
])

images = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".nii.gz")
])

data = [{"image": img} for img in images]
dataset = Dataset(data=data, transform=infer_transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

post_pred = AsDiscrete(argmax=True)

with torch.no_grad():
    for i, batch in enumerate(loader):
        img = batch["image"].to(device)
        output = model(img)
        seg = post_pred(output)[0].cpu().numpy()

        patient_id = os.path.basename(images[i]).replace(".nii.gz", "")
        out_path = os.path.join(OUT_DIR, f"{patient_id}_seg.nii.gz")

        nib.save(
            nib.Nifti1Image(seg.astype(np.uint8), np.eye(4)),
            out_path
        )

        print(f"Saved mask: {out_path}")
