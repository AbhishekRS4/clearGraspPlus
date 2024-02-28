#!/usr/bin/env python3

from __future__ import print_function, division
import os
import glob
import torch
import imageio
import numpy as np
import imgaug as ia


from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa
from pyats.datastructures import NestedAttrDict
from torch.utils.data import Dataset, DataLoader


class OcclusionBoundaryDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format). If no labels exists, pass empty string.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 input_dir='data/datasets/train/milk-bottles-train/resized-files/preprocessed-rgb-imgs',
                 transform=None,
                 input_only=None,
                 ):

        super().__init__()

        self.images_dir = input_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._extension_input = ['-rgb.png', '-rgb.jpg', '-transparent-rgb-img.jpg', 'input-img.jpg']  # The file extension of input images
        self._create_lists_filenames(self.images_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
        '''

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        return _img_tensor, image_path

    def _create_lists_filenames(self, images_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
        '''

        assert os.path.isdir(images_dir), f'Dataloader given images directory that does not exist: {images_dir}'
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths
        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError(f'No images found in given directory {images_dir}. Searched for {self._extension_input}')

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


def load_concat_sub_datasets(dataset_type, aug_transform, percent_data=None, input_only=None):
    db_list = []
    for sub_dataset in dataset_type:
        sub_dataset = NestedAttrDict(**sub_dataset)
        db_sub_dataset = OcclusionBoundaryDataset(
            input_dir=sub_dataset.images,
            transform=aug_transform,
            input_only=input_only,
        )
        if percent_data is not None:
            data_size = int(percent_data * len(db_sub_dataset))
            db_sub_dataset = torch.utils.data.Subset(db_sub_dataset, range(data_size))
        db_list.append(db_sub_dataset)

    db_complete_set = torch.utils.data.ConcatDataset(db_list)
    return db_complete_set


def get_data_loader(db_set, batch_size, num_workers=8, shuffle=False, pin_memory=False, drop_last=True):
    data_loader = DataLoader(
        db_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    return data_loader


def get_augumentation_list(which_set, img_height, img_width):
    augs_list_for_a_set = None

    if which_set == "train":
        augs_train = iaa.Sequential([
            # Geometric Augs
            iaa.Resize({
                "height": img_height,
                "width": img_width
            }, interpolation='nearest'),
            iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            # iaa.Rot90((0, 4)),

            # Bright Patches
            iaa.Sometimes(
                0.1,
                iaa.blend.Alpha(factor=(0.2, 0.7),
                                first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                                  upscale_method='cubic',
                                                                  iterations=(1, 2)),
                                name="simplex-blend")),

            # Color Space Mods
            iaa.Sometimes(
                0.3,
                iaa.OneOf([
                    iaa.Add((20, 20), per_channel=0.7, name="add"),
                    iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
                    iaa.WithColorspace(to_colorspace="HSV",
                                       from_colorspace="RGB",
                                       children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                                       name="hue"),
                    iaa.WithColorspace(to_colorspace="HSV",
                                       from_colorspace="RGB",
                                       children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                                       name="sat"),
                    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
                    iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
                ])),

            # Blur and Noise
            iaa.Sometimes(
                0.4,
                iaa.SomeOf((1, None), [
                    iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                               iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
                    iaa.OneOf([
                        iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                        iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                        iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                        iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                        iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
                    ]),
                ],
                           random_order=True)),

            # Colored Blocks
            iaa.Sometimes(0.4, iaa.CoarseDropout(0.03, size_px=(4, 8), per_channel=True, name="cdropout")),
            iaa.Sometimes(0.15, iaa.CoarseDropout(0.02, size_px=(4, 10), per_channel=False, name="cdropout_black")),
        ])
        augs_list_for_a_set = augs_train
    elif which_set == "validation" or which_set == "test":
        augs_test = iaa.Sequential([
            iaa.Resize({
                "height": img_height,
                "width": img_width
            }, interpolation='nearest'),
        ])
        augs_list_for_a_set = augs_test

    return augs_list_for_a_set
