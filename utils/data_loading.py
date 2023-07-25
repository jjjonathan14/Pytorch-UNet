import logging
import numpy as np
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from numpy import random
import imgaug.augmenters as iaa
# from utils.augmentations import add_dust


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, img_size: tuple = (968, 732)):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.rand_value = (random.rand(1)[0] / 10.0)
        self.augment_pixel_list = ['ColorJitter']
        self.augment_pixel_list_imgaug = []
        self.augment_pixel_custom = []  # dust
        self.augment_shape_list = ['affine', 'rotate']
        self.add_dust = False  # add dust augmentation
        self.augmentations = {
                                    'affine': lambda img: transforms.functional.affine(img, random.randint(20), (0, 0), 1, 0),
                                    'rotate': lambda img: transforms.functional.rotate(img, random.randint(180)),
                                    'hflip': lambda img: transforms.functional.hflip(img),
                                    'vflip': lambda img: transforms.functional.vflip(img),

                                    'ColorJitter': transforms.ColorJitter(self.rand_value, self.rand_value, self.rand_value, self.rand_value), # brightness , contrast , saturation , hue

                                    # 'dust': lambda img: add_dust(img),
                                    'AdditiveGaussianNoise': iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                                    'AdditiveLaplaceNoise': iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)),
                                    'GaussianBlur':iaa.GaussianBlur(sigma=(0, 10)),
                                    'AverageBlur':iaa.AverageBlur(k=(2, 11)),
                                    'MedianBlur':iaa.MedianBlur(k=(3, 11)),
                                    'GammaContrast': iaa.GammaContrast((0.5, 2.0)),
                                    'SigmoidContrast': iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
                                    'Sharpen': iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.5))
                                    }
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.mask_values = [[0, 0, 0], [255, 0, 0]]
        logging.info(f'Unique mask values: {self.mask_values}')

        self.aug_count = 1 + len(self.augment_pixel_list) + len(self.augment_pixel_list_imgaug) + len(self.augment_shape_list)
        if self.add_dust:
            self.aug_count += 1

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, img_size, is_mask):
        # w, h = pil_img.size
        newW, newH = img_size
        # pil_img = pil_img.resize((newW, newH), resample=Image.ANTIALIAS if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
            return img

    def preprocess_aug(self, pil_imgs, img_size, is_mask):
        # w, h = pil_img.size
        imgList_ndarray = []
        for pil_img in pil_imgs:
            img_ndarray = self.preprocess(self.mask_values, pil_img, img_size, is_mask)
            imgList_ndarray.append(img_ndarray)
        return imgList_ndarray

    def apply_augmentation(self, pil_image, pil_mask):
        # shape changing augmentations in images
        PT = transforms.ToTensor()
        TP = transforms.ToPILImage()
        temp_tf_augment_img = []
        temp_tf_augment_mask = []
        pt1, pt2 = PT(pil_image.copy()), PT(pil_mask.copy())
        new = torch.stack([pt1[0], pt1[1], pt1[2], pt2[0], pt2[1], pt2[2]])

        AugTransforms = [self.augmentations[aug](new.clone()) for aug in self.augment_shape_list]

        for result in AugTransforms:
            tp1, tp2 = TP(result[0:3]), TP(result[3:])
            temp_tf_augment_img.append(tp1)
            temp_tf_augment_mask.append(tp2)

        temp_tf_pixel_img = []
        temp_tf_pixel_mask = []

        # pixel change augmentation in images: pytorch
        for tf_augment_pixel in self.augment_pixel_list:
            img_temp = self.augmentations[tf_augment_pixel](pil_image.copy())
            temp_tf_pixel_img.append(img_temp)
            temp_tf_pixel_mask.append(pil_mask)

        # pixel change augmentation in images: imgaug
        for tf_augment_pixel in self.augment_pixel_list_imgaug:
            img_temp = self.augmentations[tf_augment_pixel](image=np.array(pil_image.copy()))
            img_temp = Image.fromarray(np.uint8(img_temp)).convert('RGB')
            temp_tf_pixel_img.append(img_temp)
            temp_tf_pixel_mask.append(pil_mask)

        # dust augmentation
        for aug in self.augment_pixel_custom:
            img_temp = self.augmentations[aug](np.array(pil_image.copy()))
            temp_tf_pixel_img.append(img_temp)
            temp_tf_pixel_mask.append(pil_mask)

        return temp_tf_pixel_img, temp_tf_augment_img, temp_tf_pixel_mask, temp_tf_augment_mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # augment images and masks
        # img_aug, img_augSahpe, mask_aug, mask_augSahpe = self.apply_augmentation(img.copy(), mask.copy())

        img = self.preprocess(self.mask_values, img, self.img_size, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.img_size, is_mask=True)

        # auged_img = self.preprocess_aug(img_aug, self.img_size, is_mask=False)
        # auged_mask = self.preprocess_aug(mask_aug, self.img_size, is_mask=True)

        # auged_img_shape = self.preprocess_aug(img_augSahpe, self.img_size, is_mask=False)
        # auged_mask_shape = self.preprocess_aug(mask_augSahpe, self.img_size, is_mask=True)

        return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous(),
                # 'auged_shape_image': [torch.as_tensor(x.copy()).float().contiguous() for x in auged_img_shape],
                # 'auged_shape_mask': [torch.as_tensor(x.copy()).long().contiguous() for x in auged_mask_shape],
                # 'auged_image': [torch.as_tensor(x.copy()).float().contiguous() for x in auged_img],
                # 'auged_mask': [torch.as_tensor(x.copy()).long().contiguous() for x in auged_mask],
            }

        # return torch.as_tensor(img.copy()).float().contiguous(), torch.as_tensor(mask.copy()).long().contiguous()


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, img_size=(968, 732)):
        super().__init__(images_dir, mask_dir, img_size)
