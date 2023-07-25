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

    return mask[0].long().squeeze().numpy()


def overlay_mask(image, mask, color, j, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        if j == 0:
            alpha = 0
            image[:, :, c] = np.where(mask == j,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        else:
            image[:, :, c] = np.where(mask == j,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
    return image


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
            out[mask == i] = v
        img = Image.fromarray(out)
    return img, out


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/kaggle/input/unet-1-class/unet_1_class.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images or the directory',
                        required=False, default='/kaggle/input/hubmap-hacking-the-human-vasculature/test')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', default=None,
                        help='Filenames of output images or directory')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.2,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--imsize', '-s', nargs='+', type=int, help='resize w and h of the images', default=[512, 512])
    parser.add_argument('--dropout', action='store_true', default=False, help='Use drop out layers')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--overlay', '-mo', action='store_true', default=False, help='overlay the mask on top of image')
    parser.add_argument('--unet-vgg', action='store_true', default=True, help='Load Unet VGG model architecture')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    if args.unet_vgg:
        net = UNet_VGG(out_channels=args.classes)
    else:
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, dropout=args.dropout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')


    for img in os.listdir(args.input):
        img_open = cv.imread(f'{args.input}/{img}')
        mask_out = predict_img(net=net,
                               full_img=img_open,
                               img_size=args.imsize,
                               out_threshold=args.mask_threshold,
                               device=device)
        result, out = mask_to_image(mask_out, mask_values, img_open, mask_overlay=False)
        result.save('test.jpg')
        gray = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
        edged = cv.Canny(gray, 30, 255)
        contours, hierarchy = cv.findContours(edged,
                                               cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


        print(contours)
    #
    # if os.path.isdir(in_files[0]):
    #     for im_file in os.listdir(in_files[0]):
    #         if im_file.endswith('.jpg') or im_file.endswith('.jpeg') or im_file.endswith('.png'):
    #             logging.info(f'Predicting image {im_file} ...')
    #             img_in = Image.open(f'{in_files[0]}{im_file}')
    #             mask_out = predict_img(net=net,
    #                                    full_img=img_in,
    #                                    img_size=args.imsize,
    #                                    out_threshold=args.mask_threshold,
    #                                    device=device)
    #
    #             if not args.no_save:
    #                 if not args.output:
    #                     if not os.path.exists('predictions/'):
    #                         os.makedirs('predictions/')
    #                     out_dir = 'predictions/'
    #                 else:
    #                     if not os.path.isdir(args.output[0]):
    #                         os.makedirs(args.output[0])
    #                     out_dir = args.output[0]
    #
    #                 out_filename = f'{out_dir}{".".join(im_file.split(".")[:-1])}_predicted.jpg'
    #                 img_in = np.asarray(img_in)
    #                 result = mask_to_image(mask_out, mask_values, img_in, mask_overlay=args.overlay)
    #                 result.save(out_filename)
    #                 logging.info(f'Mask saved to {out_filename}')
    #
    #             if args.viz:
    #                 logging.info(f'Visualizing results for image {im_file}, close to continue...')
    #                 plot_img_and_mask(img_in, mask_out)
    #
    # else:
    #     for i, filename in enumerate(in_files):
    #         logging.info(f'Predicting image {filename} ...')
    #         img_in = Image.open(filename)
    #
    #         mask_out = predict_img(net=net,
    #                                full_img=img_in,
    #                                img_size=args.imsize,
    #                                out_threshold=args.mask_threshold,
    #                                device=device)
    #
    #         if not args.no_save:
    #             out_filename = out_files[i]
    #             img_in = np.asarray(img_in)
    #             result = mask_to_image(mask_out, mask_values, img_in, mask_overlay=args.overlay)
    #             result.save(out_filename)
    #             logging.info(f'Mask saved to {out_filename}')
    #
    #         if args.viz:
    #             logging.info(f'Visualizing results for image {filename}, close to continue...')
    #             plot_img_and_mask(img_in, mask_out)
