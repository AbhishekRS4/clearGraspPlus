"""
script for saving occlusion boundary predictions
"""

import os
import sys
import cv2
import errno
import oyaml
import torch
import imageio
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
from pyats.datastructures import NestedAttrDict

from utils import utils
from modeling import deeplab
from dataset_for_vis import get_augumentation_list, get_data_loader, load_concat_sub_datasets


def run_model_save_predictions(model, test_loader, device, num_classes,
    dir_results_top=None,
    precision=5, eps=1):

    model.eval()
    num_images = len(test_loader.dataset)  # Num of total images

    with torch.no_grad():
        for ii, batch in enumerate(tqdm(test_loader)):
            inputs, image_path = batch

            # Forward pass of the mini-batch
            inputs = inputs.to(device, dtype=torch.float)

            outputs = model(inputs)
            output_softmax = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()
            pred_labels = torch.argmax(outputs, 1)

            #print("pred:", pred_labels.shape, pred_labels.dtype, pred_labels.min(), pred_labels.max())
            #print("label:", labels.shape, labels.dtype, labels.min(), labels.max())

            if dir_results_top is not None:
                dataset_string = image_path[0].split("/")[-3]
                image_string = image_path[0].split("/")[-1].split(".")[0]

                # save RGB viz of occlusion boundaries
                input_image = np.squeeze(inputs.detach().cpu().numpy())
                #print(input_image.shape)
                input_image = (input_image.transpose(1, 2, 0) * 255).astype(np.uint8)

                pred_labels = np.squeeze(pred_labels.detach().cpu().numpy())

                result_rgb_img = np.zeros_like(input_image)
                result_rgb_img[pred_labels == 0, 0] = 255 # R
                result_rgb_img[pred_labels == 1, 1] = 255 # G
                result_rgb_img[pred_labels == 2, 2] = 255 # B
                file_path_result_rgb = os.path.join(
                    dir_results_top,
                    f"{dataset_string}_{image_string}_occ_boundary.png"
                )
                imageio.imwrite(file_path_result_rgb, result_rgb_img)
    return


def save_prediction_vis(ARGS):
    ###################### Load Config File #############################
    FILE_PATH_CONFIG = ARGS.config_file
    with open(FILE_PATH_CONFIG) as fd_config_yaml:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print("Inference of occlusion boundary prediction model. Loading checkpoint...")

    ###################### Load Checkpoint and its data #############################
    if not os.path.isfile(config.vis.pathWeightsFile):
        raise ValueError(f"Invalid path to the given weights file in config. The file ({config.vis.pathWeightsFile}) does not exist")

    CHECKPOINT = torch.load(config.vis.pathWeightsFile, map_location="cpu")
    if "model_state_dict" in CHECKPOINT:
        print(f"Loaded data from checkpoint weights file: {config.vis.pathWeightsFile}")
    else:
        raise ValueError("The checkpoint file does not have model_state_dict in it.\
                         Please use the newer checkpoint files!")

    checkpoint_epoch_num = config.vis.pathWeightsFile.split("/")[-1].split(".")[0].split("_")[-1]

    DIR_RESULTS_ROOT = config.vis.resultsDir
    DIR_RESULTS = os.path.join(DIR_RESULTS_ROOT, f"occ_bound_epoch_{checkpoint_epoch_num}")
    if not os.path.isdir(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)

    print(f"Saving results to folder: {DIR_RESULTS}")

    ###################### DataLoader #############################
    augs_test = get_augumentation_list("test",
        config.vis.imgHeight, config.vis.imgWidth
    )

    # Make new dataloaders for each test set - real
    db_test = None
    if config.vis.datasets is not None:
        print(f"loading test real dataset from: {config.vis.datasets}")
        db_test = load_concat_sub_datasets(
            config.vis.datasets,
            augs_test,
            percent_data=None,
            input_only=None,
        )

    if (db_test is None):
        raise ValueError("No valid datasets provided to run inference on!")

    if db_test:
        test_loader = get_data_loader(db_test,
                                      batch_size=config.vis.batchSize,
                                      drop_last=False)

    ###################### ModelBuilder #############################
    if config.vis.model == "drn":
        model = deeplab.DeepLab(num_classes=config.vis.numClasses, backbone="drn", sync_bn=True, freeze_bn=False)
    elif config.vis.model == "drn_psa":
        model = deeplab.DeepLab(num_classes=config.vis.numClasses, backbone="drn_psa", sync_bn=True,
                                freeze_bn=False)  # output stride is 8 for drn_psa
    elif config.vis.model == 'resnet34_psa':
        model = deeplab.DeepLab(num_classes=config.vis.numClasses, backbone='resnet34_psa', sync_bn=True,
                                freeze_bn=False)
    elif config.vis.model == 'resnet50_psa':
        model = deeplab.DeepLab(num_classes=config.vis.numClasses, backbone='resnet50_psa', sync_bn=True,
                                freeze_bn=False)
    else:
        raise ValueError(f"Invalid model ({config.vis.model}) in config file")

    model.load_state_dict(CHECKPOINT["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dict_dataset_loader = {}
    if db_test:
        dict_dataset_loader.update({"real": test_loader})

    ### Save the Test Set predictions ###
    print("\nInference - occlusion boundary prediction task")
    print("=" * 50 + "\n")

    dir_results_top = DIR_RESULTS

    for key in dict_dataset_loader:
        print("\n" + key + ":")
        print("=" * 30)

        test_loader_current = dict_dataset_loader[key]
        num_images_current = len(test_loader_current.dataset)

        if num_images_current == 0:
            continue

        print(f"test set: {key}, num images: {num_images_current}")
        run_model_save_predictions(
            model,
            test_loader_current,
            device,
            config.vis.numClasses,
            dir_results_top=dir_results_top,
        )
    return


def main():
    parser = argparse.ArgumentParser(
        description="Save predictions of occlusion boundary seg model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config_file", required=True, help="Path to yaml config file", metavar="path/to/config.yaml")
    ARGS = parser.parse_args()
    save_prediction_vis(ARGS)
    return


if __name__ == "__main__":
    main()
