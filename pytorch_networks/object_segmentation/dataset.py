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


class ObjectSegmentationDataset(Dataset):
    """
    Dataset class for training model on estimation of surface normals.
    Uses imgaug for image augmentations.

    If a label_dir is blank ( None, ""), it will assume labels do not exist and return a tensor of zeros
    for the label.

    Args:
        input_dir (str): Path to folder containing the input images (.png format).
        label_dir (str): (Optional) Path to folder containing the labels (.png format). If no labels exists, pass empty string.
        transform (imgaug transforms): imgaug Transforms to be applied to the imgs
        input_only (list, str): List of transforms that are to be applied only to the input img

    """

    def __init__(self,
                 input_dir="data/datasets/train/milk-bottles-train/resized-files/preprocessed-rgb-imgs",
                 label_dir="",
                 transform=None,
                 input_only=None,
                 ):

        super().__init__()

        self.images_dir = input_dir
        self.labels_dir = label_dir
        self.transform = transform
        self.input_only = input_only

        # Create list of filenames
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_label = []  # Variable containing list of all ground truth filenames in dataset
        self._extension_input = ["-transparent-rgb-img.jpg", "-rgb.jpg"]  # The file extension of input images
        self._extension_label = "-mask.png"  # The file extension of labels
        self._create_lists_filenames(self.images_dir, self.labels_dir)

    def __len__(self):
        return len(self._datalist_input)

    def __getitem__(self, index):
        """Returns an item from the dataset at the given index. If no labels directory has been specified,
        then a tensor of zeroes will be returned as the label.

        Args:
            index (int): index of the item required from dataset.

        Returns:
            torch.Tensor: Tensor of input image
            torch.Tensor: Tensor of label
        """

        # Open input imgs
        image_path = self._datalist_input[index]
        _img = Image.open(image_path).convert("RGB")
        _img = np.array(_img)

        # Open labels
        if self.labels_dir:
            label_path = self._datalist_label[index]
            mask = imageio.imread(label_path)
            _label = np.zeros(mask.shape, dtype=np.uint8)
            _label[mask >= 100] = 1

        # Apply image augmentations and convert to Tensor
        if self.transform:
            det_tf = self.transform.to_deterministic()
            _img = det_tf.augment_image(_img)
            _img = np.ascontiguousarray(_img)  # To prevent errors from negative stride, as caused by fliplr()
        if self.labels_dir:
            _label = det_tf.augment_image(_label, hooks=ia.HooksImages(activator=self._activator_masks))

        # Return Tensors
        _img_tensor = transforms.ToTensor()(_img)
        if self.labels_dir:
            _label_tensor = torch.from_numpy(_label.astype(np.float32))
            _label_tensor = torch.unsqueeze(_label_tensor, 0)
            # _label_tensor = transforms.ToTensor()(_label.astype(np.float))
        else:
            _label_tensor = torch.zeros((1, _img_tensor.shape[1], _img_tensor.shape[2]), dtype=torch.float32)

        return _img_tensor, _label_tensor

    def _create_lists_filenames(self, images_dir, labels_dir):
        """Creates a list of filenames of images and labels each in dataset
        The label at index N will match the image at index N.

        Args:
            images_dir (str): Path to the dir where images are stored
            labels_dir (str): Path to the dir where labels are stored

        Raises:
            ValueError: If the given directories are invalid
            ValueError: No images were found in given directory
            ValueError: Number of images and labels do not match
        """

        assert os.path.isdir(images_dir), f"Dataloader given images directory that does not exist: {images_dir}"
        for ext in self._extension_input:
            imageSearchStr = os.path.join(images_dir, "*" + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths
        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError("No images found in given directory. Searched for {}".format(imageSearchStr))

        if labels_dir:
            assert os.path.isdir(labels_dir), (f"Dataloader given labels directory that does not exist: {labels_dir}")
            labelSearchStr = os.path.join(labels_dir, "*" + self._extension_label)
            labelpaths = sorted(glob.glob(labelSearchStr))
            self._datalist_label = labelpaths
            numLabels = len(self._datalist_label)
            if numLabels == 0:
                raise ValueError(f"No labels found in given directory. Searched for {imageSearchStr}")
            if numImages != numLabels:
                raise ValueError(f"The number of images and labels do not match. Please check data,\
                                found {numImages} images and {numLabels} labels")

    def _activator_masks(self, images, augmenter, parents, default):
        """Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        """
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default


def load_concat_sub_datasets(dataset_type, aug_transform, percent_data=None, input_only=None):
    db_list = []
    for sub_dataset in dataset_type:
        sub_dataset = NestedAttrDict(**sub_dataset)
        db_sub_dataset = ObjectSegmentationDataset(
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
                "height": img_height, # replace with img_height
                "width": img_width # replace with img_width
            }, interpolation="nearest"),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90((0, 4)),

            # Bright Patches
            iaa.Sometimes(
                0.1,
                iaa.blend.Alpha(factor=(0.2, 0.7),
                                first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                                  upscale_method="cubic",
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
                "height": img_height, # replace with img_height
                "width": img_width # replace with img_width
            }, interpolation="nearest"),
        ])
        augs_list_for_a_set = augs_test

    return augs_list_for_a_set


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import transforms
    import torchvision

    augs = None  # augs_train, augs_test, None
    input_only = None  # ["gaus-blur", "grayscale", "gaus-noise", "brightness", "contrast", "hue-sat", "color-jitter"]

    db_test = ObjectSegmentationDataset(
        input_dir="data/datasets/milk-bottles/resized-files/preprocessed-rgb-imgs",
        label_dir="data/datasets/milk-bottles/resized-files/preprocessed-camera-normals",
        transform=augs,
        input_only=input_only
    )

    batch_size = 16
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)

    # Show 1 Shuffled Batch of Images
    for ii, batch in enumerate(testloader):
        # Get Batch
        img, label = batch
        print("image shape, type: ", img.shape, img.dtype)
        print("label shape, type: ", label.shape, label.dtype)

        # Show Batch
        sample = torch.cat((img, label), 2)
        im_vis = torchvision.utils.make_grid(sample, nrow=batch_size // 4, padding=2, normalize=True, scale_each=True)
        plt.imshow(im_vis.numpy().transpose(1, 2, 0))
        plt.show()

        break
