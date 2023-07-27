
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib

def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str



import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.data_loading import BasicDataset
from unet import UNet
from unet.unet_vgg_model import UNet_VGG
from utils.utils import plot_img_and_mask
from unet.unet_vgg_model import UNet_VGG
import cv2 as cv
import pandas as pd

def predict_img(net,
                full_img,
                device,
                img_size=(968, 732),
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, img_size=img_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        mask = output.argmax(dim=1)
        mask = mask > out_threshold
    return mask.float().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values, img, mask_overlay):
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    if mask_overlay:
        for i, v in enumerate(mask_values):
            img = overlay_mask(img, mask, v, i)
        img = Image.fromarray(img)
    else:
        if isinstance(mask_values[0], list):
            out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
        elif mask_values == [0, 1]:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
        else:
            out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
        for i, v in enumerate(mask_values):
            if i == 1:
                v = [255,255,255]
            out[mask == i] = v
        img = Image.fromarray(out)
    return img, out



in_files = '/kaggle/input/hubmap-hacking-the-human-vasculature/test'

net = UNet_VGG(out_channels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net.to(device=device)
state_dict = torch.load('/kaggle/input/unet-1-class-36/unet_1_class_36.pth', map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
ids = []
heights = []
widths = []
prediction_strings = []

for img in os.listdir(in_files):

    img_open = Image.open(f'{in_files}/{img}')
    img_open_cv = cv.imread(f'{in_files}/{img}')
    pred = predict_img(net=net,
                       full_img=img_open,
                       img_size=(512 ,512),
                       out_threshold=0.1,
                       device=device)

    pred_string = ''
    kernel = np.ones(shape=(3, 3), dtype=np.uint8)
    binary_mask = cv.dilate((pred * 255), kernel, 3)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask.astype(np.uint8))
    print(num_labels)
    for i in range(1, num_labels):
        mask_i = np.zeros_like(binary_mask)
        mask_i[labels == i] = 1
        mask = mask_i[:, :, np.newaxis].astype(np.bool)
        score = 1.0
        encoded = encode_binary_mask(mask)
        if i== 0:
            pred_string += f"0 {score} {encoded.decode('utf-8')}"
        else:
            pred_string += f" 0 {score} {encoded.decode('utf-8')}"

    h, w, _ = img_open_cv.shape
    ids.append(img.split('.')[0])
    heights.append(h)
    widths.append(w)
    prediction_strings.append(pred_string)
os.chdir('/kaggle/working/')
sub = pd.DataFrame({"id": ids, "height": heights, "width": widths, "prediction_string": prediction_strings})
sub = sub.set_index("id")

