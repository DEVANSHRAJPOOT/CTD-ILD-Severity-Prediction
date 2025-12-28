import glob
import os

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
)
from monai.data import Dataset, DataLoader


def get_dataloader(
    image_dir,
    label_dir,
    batch_size=1,
    shuffle=True
):
    """
    Creates a MONAI DataLoader for CTD-ILD segmentation.
    """

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    if len(label_paths) == 0:
        raise RuntimeError(f"No labels found in {label_dir}")

    data = [
        {"image": img, "label": lbl}
        for img, lbl in zip(image_paths, label_paths)
    ]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=400,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            allow_smaller=True
        ),
    ])

    dataset = Dataset(data=data, transform=transforms)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )

    return loader
