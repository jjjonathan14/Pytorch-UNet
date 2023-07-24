import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss, UnetLoss
from utils.wandb import save_table
import gc
from random import sample
from unet.unet_vgg_model import UNet_VGG

torch.cuda.empty_cache()

dir_path = Path('./data/label')
unlabel_dir_path = Path('./data/unlabel')
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-3,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_size: tuple = (224, 224),
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        augmentations=False,
        aug_percentage=0.2,
        unet_vgg=False
):
    # 1. Create dataset
    seperate = True
    try:
        if seperate:

            dataset_train = CarvanaDataset(f'{dir_path}/train/imgs', f'{dir_path}/train/masks', img_size)
            dataset_validation = CarvanaDataset(f'{dir_path}/validation/imgs', f'{dir_path}/validation/masks', img_size)
            # dataset_test = CarvanaDataset(f'{dir_path}/test/imgs', f'{dir_path}/test/masks', img_size)
            # dataset_train_unlabel = CarvanaDataset(f'{unlabel_dir_path}/train/imgs', f'{unlabel_dir_path}/train/masks', img_size)
            dataset = dataset_train
        else:
            dataset = CarvanaDataset(dir_img, dir_mask, img_size)


    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_size)


    if seperate:
        n_val = len(dataset_validation)
        n_train = len(dataset_train)
        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        label_train_loader = DataLoader(dataset_train, shuffle=True, **loader_args)
        label_val_loader = DataLoader(dataset_validation, shuffle=False, drop_last=True, **loader_args)
        # label_test_loader = DataLoader(dataset_test, shuffle=False, drop_last=True, **loader_args)
        # unlabel_train_loader = train_loader = DataLoader(dataset_train_unlabel, shuffle=True, **loader_args)
        total_batch_size = len(label_train_loader)
    else:
        aug_count = dataset.aug_count
        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
        total_batch_size = len(train_loader)
    # 3. Create data loaders


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_size}
        Mixed Precision: {amp}
    ''')

    
    if unet_vgg:

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = UnetLoss
        count_aug = int(total_batch_size * aug_percentage) * 4
        count_non_aug = total_batch_size - int(total_batch_size * aug_percentage)
        total_aug_count = count_aug + count_non_aug

        # (Initialize logging)
        WANDB_CONFIG = {'_wandb_kernel': 'neuracort'}
        # Initialize W&B
        run = wandb.init(
            project='semantic_segmentation_unet',
            config=WANDB_CONFIG
        )

    else:
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0

        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      val_percent=val_percent, save_checkpoint=save_checkpoint, img_size=img_size,
                                      amp=amp))

    for epoch in range(1, epochs + 1):

        val_loader = label_val_loader
        train_loader = label_train_loader
        total_batch_size = len(train_loader)
        # if epoch % 3 != 0:
        #     train_loader = label_train_loader
        #     total_batch_size = len(train_loader)
        # else:
        #     train_loader = unlabel_train_loader
        #     total_batch_size = len(train_loader)

        selected_batches = sorted(sample(range(0, total_batch_size), int(total_batch_size * aug_percentage)))
        train_loss, train_acc = 0, 0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            aug_batch_list_images, aug_batch_list_masks = [batch['image']], [batch['mask']]
            if augmentations and i in selected_batches:
                # augmentations which is not changing the image shapes
                for img_ in batch['auged_image']:
                    aug_batch_list_images.append(img_)

                # augmentations which is changing the image shape/perception
                for img_ in batch['auged_shape_image']:
                    aug_batch_list_images.append(img_)

                # augmentations which is not changing the mask shapes
                for mask_ in batch['auged_mask']:
                    aug_batch_list_masks.append(mask_)

                # augmentations which is changing the mask shape/perception
                for mask_ in batch['auged_shape_mask']:
                    aug_batch_list_masks.append(mask_)
            del batch
            gc.collect()
            torch.cuda.empty_cache()
            for true_masks, images in zip(aug_batch_list_masks, aug_batch_list_images):
                images = images.to(device=device)
                true_masks = true_masks.to(device=device)

                if unet_vgg:
                    masks_pred = model(images)
                    model.to(device)
                    model.train()
                    optimizer.zero_grad()
                    train_loss, train_accuracy = criterion(masks_pred, true_masks)
                    train_loss.backward()
                    optimizer.step()
                    train_loss += train_loss.item()
                    train_acc += train_accuracy.item()
                else:
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        train_loss += loss.item()
                        experiment.log({
                            'train loss': loss.item(),
                            'step': global_step,
                            'epoch': epoch
                        })

            del aug_batch_list_masks
            del aug_batch_list_images
            gc.collect()

        if unet_vgg:
            val_loss, val_acc = 0, 0
            for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
                model.to(device)
                model.eval()
                images = batch['image'].to(device=device)
                true_masks = batch['mask'].to(device=device)
                masks_pred = model(images)
                del batch
                gc.collect()
                val_loss1, val_acc1 = criterion(masks_pred, true_masks)
                val_loss += val_loss1.item()
                val_acc += val_acc1.item()

            wandb.log(
                {
                    'epoch': epoch,
                    'train_loss': train_loss / total_aug_count,
                    'train_acc': train_acc / total_aug_count,
                    'val_loss': val_loss / len(val_loader),
                    'val_acc': val_acc / len(val_loader)
                }
            )
            # save_table(f'{epoch}', val_loader, model, device)

        else:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')
                if not torch.isinf(value).any():
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                if not torch.isinf(value.grad).any():
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass



        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--imsize', '-s', nargs='+', type=int, help='resize w and h of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--dropout', action='store_true', default=False, help='Use drop out layers')
    parser.add_argument('--augmentations', action='store_true', default=False, help='add or remove augmentations')
    parser.add_argument('--aug-percentage', '-ap', type=float, default=0.2, help='augmentation applying percentage')
    parser.add_argument('--unet-vgg', action='store_true', default=False, help='Load Unet VGG model architecture')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.unet_vgg:
        model = UNet_VGG(out_channels=args.classes).to(device)
    else:
        model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, dropout=args.dropout)
        model = model.to(memory_format=torch.channels_last)
        model.to(device=device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.imsize,
            val_percent=args.val / 100,
            amp=args.amp,
            augmentations=args.augmentations,
            aug_percentage=args.aug_percentage,
            unet_vgg=args.unet_vgg
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_size=args.imsize,
            val_percent=args.val / 100,
            amp=args.amp,
            augmentations=args.augmentations,
            aug_percentage=args.aug_percentage,
            unet_vgg=args.unet_vgg
        )