from genericpath import isdir, isfile
import random
from unittest import loader
import torch.utils.data as data
import numpy as np
import os
import glob
from PIL import Image
import torchvision.transforms.functional as transFunc
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, gt_file, mask_file, config):
        self.gt_image_files = self.load_file_list(gt_file)
        self.mask_file = mask_file

        if len(self.gt_image_files) == 0:
            raise(RuntimeError("Found 0 images in the files:\n"+gt_file))

        if config.isTrain is False:
            self.transform_cfg = {
                "crop": False,
                "flip": False,
                "resize": config.test_image_size,
                "random_load_mask": False,
            }
            config.mask_type == "from_file" if mask_file is not None else config.mask_type
        else:
            self.transform_cfg = {
                "crop": config.need_crop,
                "flip": config.need_flip,
                "resize": config.train_image_size,
                "random_load_mask": True,
            }

        self.mask_image_file = self.load_file_list(mask_file)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print("loading error: "+self.gt_image_files[index])
            item = self.load_item(0)
        return item

    def __len__(self):
        return len(self.gt_image_files)

    def load_file_list(self, path):
        if isinstance(path, list):
            return path

        if isinstance(path, str):
            if os.path.isdir(path):
                gtList = list(glob.glob(path + "/*.jpg")) + list(
                    glob.glob(path + "/*.png")
                )
                gtList.sort()
                return gtList
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding="utf-8")
                except:
                    return [path]
        return []

    def load_item(self, index):
        gt_path = self.gt_image_files[index]
        gt_image = loader(gt_path)
        transform_param = get_params(gt_image.size, self.transform_cfg)
        gt_image = transform_image(transform_param, gt_image)

        mask = self.load_mask(index, gt_image)
        return gt_image, mask

    def load_mask(self, index,image):
        _, w, h = image.shape
        image_shape = [w, h]
        #
        
        mask = gray_loader(self.mask_file)
        mask = transFunc.resize(mask, size=image_shape)
        mask = transFunc.to_tensor(mask)
        mask = (mask > 0).float()
        return mask

        # mask = Image.open(self.mask_file)
        # mask = transFunc.resize(mask, size=image_shape)
        # mask = transFunc.to_tensor(mask)
        # mask = (mask > 0).float()


##
def get_params(size, transform_cfg):
    w, h = size
    if transform_cfg["flip"]:
        flip = random.random() > 0.5
    else:
        flip = False
    if transform_cfg["crop"]:
        transform_crop = (
            transform_cfg["crop"]
            if w >= transform_cfg["crop"][0] and h >= transform_cfg["crop"][1]
            else [h, w]
        )
        x = random.randint(0, np.maximum(0, w - transform_crop[0]))
        y = random.randint(0, np.maximum(0, h - transform_crop[1]))
        crop = [x, y, transform_crop[0], transform_crop[1]]
    else:
        crop = False
    if transform_cfg["resize"]:
        resize = [
            transform_cfg["resize"],
            transform_cfg["resize"],
        ]
    else:
        resize = False
    param = {"crop": crop, "flip": flip, "resize": resize}
    return param


def transform_image(transform_param, gt_image, normalize=True, toTensor=True):
    transform_list = []

    if transform_param["crop"]:
        crop_position = transform_param["crop"][:2]
        crop_size = transform_param["crop"][2:]
        transform_list.append(
            transforms.Lambda(lambda img: __crop(
                img, crop_position, crop_size))
        )
    if transform_param["resize"]:
        transform_list.append(transforms.Resize(transform_param["resize"]))
    if transform_param["flip"]:
        transform_list.append(transforms.Lambda(
            lambda img: __flip(img, True)))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    trans = transforms.Compose(transform_list)
    gt_image = trans(gt_image)
    return gt_image


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def loader(path):
    return Image.open(path).convert("RGB")

def gray_loader(path):
    return Image.open(path)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
