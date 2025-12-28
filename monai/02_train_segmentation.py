import os
import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

from dataset import get_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_dataloader(
    "../data/images",
    "../data/labels",
    batch_size=1
)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)

post_pred = AsDiscrete(argmax=True)
post_label = AsDiscrete(to_onehot=5)

epochs = 10

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        outputs = [post_pred(i) for i in decollate_batch(outputs)]
        labels = [post_label(i) for i in decollate_batch(labels)]
        dice_metric(y_pred=outputs, y=labels)

    dice = dice_metric.aggregate().item()
    dice_metric.reset()

    print(f"Epoch {epoch+1}/{epochs}, Loss={epoch_loss:.4f}, Dice={dice:.4f}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "unet_ctd_ild.pth")
torch.save(model.state_dict(), MODEL_PATH)

