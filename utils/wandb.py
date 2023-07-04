import matplotlib.pyplot as plt
import cv2
import wandb
import torch
from tqdm import tqdm

wandb.login()

# wandb config
WANDB_CONFIG = {'_wandb_kernel': 'neuracort'}

# Initialize W&B
run = wandb.init(
    project='semantic_segmentation_unet',
    config=WANDB_CONFIG
)


def save_table(table_name, val_dl, model, device):
    table = wandb.Table(columns=['Original Image', 'Original Mask', 'Predicted Mask'], allow_mixed_types=True)

    model.to(device)
    model.eval()
    for i, batch in tqdm(enumerate(val_dl), total=len(val_dl)):
        images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        true_masks = batch['mask'].to(device=device, dtype=torch.long)
        im, mask = [images, true_masks]
        im, mask = im.to(device), mask.to(device)
        _mask = model(im)
        _, _mask = torch.max(_mask, dim=1)

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(im[0].permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_image.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("original_mask.jpg")
        plt.close()

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(_mask.permute(1, 2, 0).detach().cpu()[:, :, 0])
        plt.savefig("predicted_mask.jpg")
        plt.close()

        table.add_data(
            wandb.Image(cv2.cvtColor(cv2.imread("original_image.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("original_mask.jpg"), cv2.COLOR_BGR2RGB)),
            wandb.Image(cv2.cvtColor(cv2.imread("predicted_mask.jpg"), cv2.COLOR_BGR2RGB))
        )

    wandb.log({table_name: table})
