'''Contains utility functions used by train/eval code.
'''
import csv
import torch
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np


def label_to_rgb(label):
    '''Output RGB visualizations of the outlines' labels

    The labels of outlines have 3 classes: Background, Depth Outlines, Surface Normal Outlines which are mapped to
    Red, Green and Blue respectively.

    Args:
        label (torch.Tensor): Shape: (batchSize, 1, height, width). Each pixel contains an int with value of class.

    Returns:
        torch.Tensor: Shape (no. of images, 3, height, width): RGB representation of the labels
    '''
    rgbArray = None
    if len(label.shape) == 4:
        # Shape: (batchSize, 1, height, width)
        rgbArray = torch.zeros((label.shape[0], 3, label.shape[2], label.shape[3]), dtype=torch.float)
        rgbArray[:, 0, :, :][label[:, 0, :, :] == 0] = 1
        rgbArray[:, 1, :, :][label[:, 0, :, :] == 1] = 1
        rgbArray[:, 2, :, :][label[:, 0, :, :] == 2] = 1
    if len(label.shape) == 3:
        # Shape: (1, height, width)
        rgbArray = torch.zeros((3, label.shape[1], label.shape[2]), dtype=torch.float)
        rgbArray[0, :, :][label[0, :, :] == 0] = 1
        rgbArray[1, :, :][label[0, :, :] == 1] = 1
        rgbArray[2, :, :][label[0, :, :] == 2] = 1
    if len(label.shape) == 2:
        # Shape: (height, width)
        rgbArray = torch.zeros((3, label.shape[0], label.shape[1]), dtype=torch.float)
        rgbArray[0, :, :][label[:, :] == 0] = 1
        rgbArray[1, :, :][label[:, :] == 1] = 1
        rgbArray[2, :, :][label[:, :] == 2] = 1

    return rgbArray


def create_grid_image(inputs, outputs, labels, max_num_images_to_save=3):
    '''Make a grid of images for display purposes
    Size of grid is (3, N, 3), where each coloum belongs to input, output, label resp

    Args:
        inputs (Tensor): Batch Tensor of shape (B x C x H x W)
        outputs (Tensor): Batch Tensor of shape (B x H x W)
        labels (Tensor): Batch Tensor of shape (B x C x H x W)
        max_num_images_to_save (int, optional): Defaults to 3. Out of the given tensors, chooses a
            max number of imaged to put in grid

    Returns:
        numpy.ndarray: A numpy array with of input images arranged in a grid
    '''

    img_tensor = inputs[:max_num_images_to_save]
    output_tensor = torch.unsqueeze(torch.max(outputs[:max_num_images_to_save], 1)[1].float(), 1)
    output_tensor_rgb = label_to_rgb(output_tensor)
    label_tensor = labels[:max_num_images_to_save]
    label_tensor_rgb = label_to_rgb(label_tensor)

    images = torch.cat((img_tensor, output_tensor_rgb, label_tensor_rgb), dim=3)
    grid_image = make_grid(images, 1, normalize=True, scale_each=True)

    return grid_image


def cross_entropy2d(logit, target, ignore_index=255, weight=None, batch_average=True):
    """
    The loss is

    .. math::
        \sum_{i=1}^{\\infty} x_{i}

        `(minibatch, C, d_1, d_2, ..., d_K)`

    Args:
        logit (Tensor): Output of network
        target (Tensor): Ground Truth
        ignore_index (int, optional): Defaults to 255. The pixels with this labels do not contribute to loss
        weight (List, optional): Defaults to None. Weight assigned to each class
        batch_average (bool, optional): Defaults to True. Whether to consider the loss of each element in the batch.

    Returns:
        Float: The value of loss.
    """

    n, c, h, w = logit.shape
    target = target.squeeze(1)

    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')

    loss = criterion(logit, target.long())

    if batch_average:
        loss /= n

    return loss


def FocalLoss(logit, target, weight, gamma=2, alpha=0.5):
    #n, c, h, w = logit.shape
    target = target.squeeze(1)
    criterion = nn.CrossEntropyLoss(weight=weight)

    logpt = -criterion(logit, target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt

    '''
    if batch_average:
        loss /= n
    '''

    return loss


# compute mean IOU
def compute_mean_IOU(true_label, pred_label, num_classes=5):
    iou_list = list()
    present_iou_list = list()

    pred_label = pred_label.view(-1)
    true_label = true_label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # in computation of IoU.
    for sem_class in range(num_classes):
        pred_label_inds = (pred_label == sem_class)
        target_inds = (true_label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float("nan")
        else:
            intersection_now = (pred_label_inds[target_inds]).long().sum().item()
            union_now = pred_label_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    present_iou_list = np.array(present_iou_list)
    return np.nanmean(present_iou_list)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    per_class_iou = [0] * n_classes
    num_images_per_class = [0] * n_classes
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        iou_per_class = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2)
            un = torch.sum(match > 0)

            intersect[j] += it
            union[j] += un

            if union[j] == 0:
                iou_per_class[j] = -1

            else:
                iou_per_class[j] = intersect[j] / union[j]
                # print('IoU for class %d is %f'%(j, iou_per_class[j]))

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        for k in range(n_classes):
            if iou_per_class[k] == -1:
                continue
            else:
                per_class_iou[k] += iou_per_class[k]
                num_images_per_class[k] += 1

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou
    # print('class iou:', per_class_iou)
    # print('images per class:', num_images_per_class)
    return total_iou, per_class_iou, num_images_per_class

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

class CSVWriter:
    """
    for writing tabular data to a csv file
    """
    def __init__(self, file_name, column_names):
        self.file_name = file_name
        self.column_names = column_names

        self.file_handle = open(self.file_name, "w")
        self.writer = csv.writer(self.file_handle)

        self.write_header()
        print(f"{self.file_name} created successfully with header row")

    def write_header(self):
        """
        writes header into csv file
        """
        self.write_row(self.column_names)
        return

    def write_row(self, row):
        """
        writes a row into csv file
        """
        self.writer.writerow(row)
        return

    def close(self):
        """
        close the file
        """
        self.file_handle.close()
        return
