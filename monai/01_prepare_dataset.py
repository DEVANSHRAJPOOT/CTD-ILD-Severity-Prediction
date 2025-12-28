from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ScaleIntensityRanged,
    RandCropByPosNegLabeld
)
from monai.data import Dataset, DataLoader
import glob
import os


def get_dataloader(image_dir, label_dir, batch_size=2):
    images = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    labels = sorted(glob.glob(os.path.join(label_dir, "*.nii*")))

    data = [{"image": i, "label": l} for i, l in zip(images, labels)]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0, clip=True
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=4
        )
    ])

    dataset = Dataset(data=data, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
