#!/usr/bin/env python3

import os
import glob
import torch
import imageio
import numpy as np
import imgaug as ia
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa
from pyats.datastructures import NestedAttrDict
from torch.utils.data import Dataset, DataLoader


from utils.utils import exr_loader


class SurfaceNormalsDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ''), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format).
                         If no labels exists, pass empty string ('') or None.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(
            self,
            input_dir,
            label_dir='',
            mask_dir='',
            transform=None,
            input_only=None,
    ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.masks_dir = mask_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []
        self._datalist_mask = []
        self._extension_input = ['-transparent-rgb-img.jpg', '-rgb.jpg', '-input-img.jpg']  # The file extension of input images
        self._extension_label = ['-cameraNormals.exr', '-normals.exr']
        self._extension_mask = ['-mask.png']
        self._create_lists_filenames(self.images_dir, self.labels_dir, self.masks_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        '''Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label (Tensor of zeroes is labels_dir is "" or None)
        '''

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert('RGB')
        _img = np.array(_img)

        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            _label = exr_loader(label_path, ndim=3)  # (3, H, W)

        if self.masks_dir:
            mask_path = self._datalist_mask[index]
            _mask = imageio.imread(mask_path)

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()

            _img = det_tf.augment_image(_img)
            if self.labels_dir:
                # Making all values of invalid pixels marked as -1.0 to 0.
                # In raw data, invalid pixels are marked as (-1, -1, -1) so that on conversion to RGB they appear black.
                mask = np.all(_label == -1.0, axis=0)
                _label[:, mask] = 0.0

                _label = _label.transpose((1, 2, 0))  # To Shape: (H, W, 3)
                _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))
                _label = _label.transpose((2, 0, 1))  # To Shape: (3, H, W)

            if self.masks_dir:
                _mask = det_tf.augment_image(_mask, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)

        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label)
            _label_tensor = nn.functional.normalize(_label_tensor, p=2, dim=0)
        else:
            _label_tensor = torch.zeros((3, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        if self.masks_dir:
            _mask = _mask[..., np.newaxis]
            _mask_tensor = transforms.ToTensor()(_mask)
        else:
            _mask_tensor = torch.ones((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor, _mask_tensor, image_path

    def _create_lists_filenames(self, images_dir, labels_dir, masks_dir):
        '''Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored
            labels_dir (str): Path to the dir where masks are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        '''

        assert os.path.isdir(images_dir), 'Dataloader given images directory that does not exist: "%s"' % (images_dir)
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))

        if labels_dir:
            assert os.path.isdir(labels_dir), ('Dataloader given labels directory that does not exist: "%s"' %
                                               (labels_dir))
            for ext in self._extension_label:
                labelSearchStr = os.path.join(labels_dir, '*' + ext)
                labelpaths = sorted(glob.glob(labelSearchStr))
                self._datalist_label = self._datalist_label + labelpaths

            numLabels = len(self._datalist_label)
            if numLabels == 0:
                raise ValueError('No labels found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numLabels:
                raise ValueError('The number of images and labels do not match. Please check data,' +
                                 'found {} images and {} labels in dirs:\n'.format(numImages, numLabels) +
                                 'images: {}\nlabels: {}\n'.format(images_dir, labels_dir))
        if masks_dir:
            assert os.path.isdir(masks_dir), ('Dataloader given masks directory that does not exist: "%s"' %
                                               (masks_dir))
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(masks_dir, '*' + ext)
                maskpaths = sorted(glob.glob(maskSearchStr))
                self._datalist_mask = self._datalist_mask + maskpaths

            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError('No masks found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numMasks:
                raise ValueError('The number of images and masks do not match. Please check data,' +
                                 'found {} images and {} masks in dirs:\n'.format(numImages, numMasks) +
                                 'images: {}\nmasks: {}\n'.format(images_dir, masks_dir))

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


# Resize Tensor
def resize_tensor(input_tensor, height, width):
    augs_label_resize = iaa.Sequential([iaa.Resize({"height": height, "width": width}, interpolation='nearest')])
    det_tf = augs_label_resize.to_deterministic()
    input_tensor = input_tensor.numpy().transpose(0, 2, 3, 1)
    resized_array = det_tf.augment_images(input_tensor)
    resized_array = torch.from_numpy(resized_array.transpose(0, 3, 1, 2))
    resized_array = resized_array.type(torch.DoubleTensor)

    return resized_array


def load_concat_sub_datasets(dataset_type, aug_transform, percent_data=None, input_only=None):
    db_list = []
    for sub_dataset in dataset_type:
        sub_dataset = NestedAttrDict(**sub_dataset)
        db_sub_dataset = SurfaceNormalsDataset(
            input_dir=sub_dataset.images,
            label_dir=sub_dataset.labels,
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
            # iaa.Fliplr(0.5),
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
                0.2,
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
            iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    # Example Augmentations using imgaug
    # imsize = 512
    # augs_train = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0), # Resize image
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Rot90((0, 4)),
    #     # Blur and Noise
    #     #iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 1.5), name="gaus-blur")),
    #     #iaa.Sometimes(0.1, iaa.Grayscale(alpha=(0.0, 1.0), from_colorspace="RGB", name="grayscale")),
    #     iaa.Sometimes(0.2, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255), per_channel=True, name="gaus-noise")),
    #     # Color, Contrast, etc.
    #     #iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25), per_channel=0.1, name="brightness")),
    #     iaa.Sometimes(0.2, iaa.GammaContrast((0.7, 1.3), per_channel=0.1, name="contrast")),
    #     iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-20, 20), name="hue-sat")),
    #     #iaa.Sometimes(0.3, iaa.Add((-20, 20), per_channel=0.5, name="color-jitter")),
    # ])
    # augs_test = iaa.Sequential([
    #     # Geometric Augs
    #     iaa.Scale((imsize, imsize), 0),
    # ])

    augs = None  # augs_train
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = SurfaceNormalsDataset(input_dir='data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs',
                                    label_dir='data/datasets/milk-bottles/resized-files/preprocessed-camera-normals',
                                    transform=augs,
                                    input_only=input_only)

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print('image shape, type: ', img.shape, img.dtype)
        print('label shape, type: ', label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.show()

        break
