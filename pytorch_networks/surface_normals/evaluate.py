"""
Evaluation script for surface normal prediction task
"""

import os
import cv2
import sys
import errno
import oyaml
import torch
import imageio
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision.utils import make_grid
from pyats.datastructures import NestedAttrDict

from utils import utils
from modeling import deeplab
from loss_functions import *
from dataset import get_augumentation_list, get_data_loader, load_concat_sub_datasets


def evaluate(model, test_loader, device, num_classes,
    dir_results_top=None, dir_sub_results=None, dir_sub_normals_rgb=None,
    dir_sub_normals_exr=None, precision=5):

    model.eval()

    running_loss_mean = 0.0
    running_loss_median = 0.0
    running_percent_1 = 0.0
    running_percent_2 = 0.0
    running_percent_3 = 0.0

    num_images = len(test_loader.dataset)  # Num of total images

    with torch.no_grad():
        for ii, batch in enumerate(tqdm(test_loader)):
            # NOTE: In raw data, invalid surface normals are represented by [-1, -1, -1]. However, this causes
            #       problems during normalization of vectors. So they are represented as [0, 0, 0] in our dataloader output.

            inputs, labels, masks, image_path = batch

            # Forward pass of the mini-batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            normal_vectors = model(inputs)
            normal_vectors_norm = nn.functional.normalize(normal_vectors, p=2, dim=1)
            normal_vectors_norm = normal_vectors_norm.detach().cpu()
            labels = labels.detach().cpu()
            mask_tensor = masks.squeeze(1)

            loss_deg_mean, loss_deg_median, percent_1, percent_2, percent_3 = \
                metric_calculator_batch(normal_vectors_norm, labels.double(), mask_tensor)

            running_loss_mean += loss_deg_mean
            running_loss_median += loss_deg_median
            running_percent_1 += percent_1
            running_percent_2 += percent_2
            running_percent_3 += percent_3

            if dir_results_top is not None:
                dataset_string = image_path[0].split("/")[-3]
                image_string = image_path[0].split("/")[-1].split(".")[0]

                #print(inputs.shape, normal_vectors_norm.shape, labels.shape, mask_tensor.shape)

                # save resulting grid image with input, prediction and label
                masks_3d = torch.stack((mask_tensor, mask_tensor, mask_tensor), dim=0)
                """
                print(inputs.squeeze().detach().cpu().shape,
                    normal_vectors_norm.squeeze().detach().cpu().shape,
                    labels.squeeze().detach().cpu().shape,
                    masks_3d.squeeze().shape
                )
                """
                grid_image = make_grid(
                    [
                        inputs.squeeze().detach().cpu(),
                        normal_vectors_norm.squeeze().detach().cpu(),
                        labels.squeeze().detach().cpu(),
                        masks_3d.squeeze()
                    ],
                    4, normalize=True, scale_each=True)

                numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
                numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)

                file_path_result = os.path.join(
                    dir_results_top, dir_sub_results,
                    f"{dataset_string}_{image_string}_sur_normals_result.png"
                )
                imageio.imwrite(file_path_result, numpy_grid)

                output_rgb = utils.normal_to_rgb(
                    normal_vectors_norm.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
                )
                output_rgb *= 255
                output_rgb = output_rgb.astype(np.uint8)
                output_rgb = cv2.resize(output_rgb, (512, 288), interpolation=cv2.INTER_LINEAR)
                file_path_normal_rgb = os.path.join(
                    dir_results_top, dir_sub_normals_rgb,
                    f"{dataset_string}_{image_string}_sur_normals_rgb.png"
                )
                file_path_normal_exr = os.path.join(
                    dir_results_top, dir_sub_normals_exr,
                    f"{dataset_string}_{image_string}_sur_normal.exr"
                )
                imageio.imwrite(file_path_normal_rgb, output_rgb)
                utils.exr_saver(file_path_normal_exr, normal_vectors_norm.squeeze().detach().cpu().numpy())
                #sys.exit(0)

    loss_mean = running_loss_mean / num_images
    loss_median = running_loss_median / num_images
    percent_1 = running_percent_1 / num_images
    percent_2 = running_percent_2 / num_images
    percent_3 = running_percent_3 / num_images

    return (loss_mean.cpu().detach().numpy(),
            loss_median.cpu().detach().numpy(),
            percent_1.cpu().detach().numpy(),
            percent_2.cpu().detach().numpy(),
            percent_3.cpu().detach().numpy(),)


def start_evaluation(ARGS):
    ###################### Load Config File #############################
    FILE_PATH_CONFIG = ARGS.config_file
    with open(FILE_PATH_CONFIG) as fd_config_yaml:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print("Inference of surface normal prediction model. Loading checkpoint...")

    ###################### Load Checkpoint and its data #############################
    if not os.path.isfile(config.eval.pathWeightsFile):
        raise ValueError(f"Invalid path to the given weights file in config. The file ({config.eval.pathWeightsFile}) does not exist")

    CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location="cpu")
    if "model_state_dict" in CHECKPOINT:
        print(f"Loaded data from checkpoint weights file: {config.eval.pathWeightsFile}")
    else:
        raise ValueError("The checkpoint file does not have model_state_dict in it.\
                         Please use the newer checkpoint files!")

    # Check for results store dir
    # Create directory to save results
    SUBDIR_RESULT = "results"
    SUBDIR_NORMALS_RGB = "normals_rgb"
    SUBDIR_NORMALS_EXR = "normals_exr"
    checkpoint_epoch_num = config.eval.pathWeightsFile.split("/")[-1].split(".")[0].split("_")[-1]

    DIR_RESULTS_ROOT = config.eval.resultsDir
    DIR_RESULTS = os.path.join(DIR_RESULTS_ROOT, f"sur_normal_{checkpoint_epoch_num}")
    if not os.path.isdir(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)

    if config.eval.saveResultImages:
        try:
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_RESULT))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_NORMALS_RGB))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_NORMALS_EXR))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    print(f"Saving results to folder: {DIR_RESULTS}")
    # Create CSV File to store error metrics
    FILE_NAME_LOGS_CSV = f"computed_metrics_epoch_{checkpoint_epoch_num}.csv"
    logging_column_names = [
        "model_name",
        "dataset",
        "loss_mean",
        "loss_median",
        "percent_1",
        "percent_2",
        "percent_3"
    ]
    csv_writer = utils.CSVWriter(
        os.path.join(DIR_RESULTS, FILE_NAME_LOGS_CSV),
        logging_column_names,
    )

    ###################### DataLoader #############################
    augs_test = get_augumentation_list("test",
        config.eval.imgHeight, config.eval.imgWidth
    )

    # Make new dataloaders for each synthetic dataset
    # test set - real
    db_test_real = None
    if config.eval.datasetsReal is not None:
        print(f"loading test real dataset from: {config.eval.datasetsReal}")
        db_test_real = load_concat_sub_datasets(
            config.eval.datasetsReal,
            augs_test,
            percent_data=None,
            input_only=None,
        )

    # test set - synthetic
    db_test_syn = None
    if config.eval.datasetsSynthetic is not None:
        print(f"loading test synthetic dataset from: {config.eval.datasetsSynthetic}")
        db_test_syn = load_concat_sub_datasets(
            config.eval.datasetsSynthetic,
            augs_test,
            percent_data=None,
            input_only=None,
        )

    if (db_test_syn is None) and (db_test_real is None):
        raise ValueError("No valid datasets provided to run inference on!")

    if db_test_syn:
        test_syn_loader = get_data_loader(db_test_syn,
                                          batch_size=config.eval.batchSize,
                                          drop_last=False)

    if db_test_real:
        test_real_loader = get_data_loader(db_test_real,
                                           batch_size=config.eval.batchSize,
                                           drop_last=False)

    ###################### ModelBuilder #############################
    if config.eval.model == "drn":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="drn", sync_bn=True,
                                freeze_bn=False)
    elif config.eval.model == "drn_psa":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="drn_psa", sync_bn=True,
                                freeze_bn=False)  # output stride is 8 for drn_psa
    elif config.eval.model == 'resnet34_psa':
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet34_psa', sync_bn=True,
                                freeze_bn=False)
    elif config.eval.model == 'resnet50_psa':
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet50_psa', sync_bn=True,
                                freeze_bn=False)
    else:
        raise ValueError(f"Invalid model ({config.eval.model}) in config file")

    model.load_state_dict(CHECKPOINT["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dict_dataset_loader = {}
    if db_test_real:
        dict_dataset_loader.update({"real": test_real_loader})
    if db_test_syn:
        dict_dataset_loader.update({"synthetic": test_syn_loader})

    ### Run evaluation on the Test Set ###
    print("\nInference - surface normal prediction task")
    print("=" * 50 + "\n")

    dir_results_top = None
    dir_sub_results = None
    dir_sub_normals_rgb = None
    dir_sub_normals_exr = None
    if config.eval.saveResultImages:
        dir_results_top = DIR_RESULTS
        dir_sub_results = SUBDIR_RESULT
        dir_sub_normals_rgb = SUBDIR_NORMALS_RGB
        dir_sub_normals_exr = SUBDIR_NORMALS_EXR

    for key in dict_dataset_loader:
        print("\n" + key + ":")
        print("=" * 30)

        test_loader_current = dict_dataset_loader[key]
        num_images_current = len(test_loader_current.dataset)

        if num_images_current == 0:
            continue

        print(f"test set: {key}, num images: {num_images_current}")
        loss_mean, loss_median, percent_1, percent_2, percent_3 = evaluate(
            model,
            test_loader_current,
            device,
            config.eval.numClasses,
            dir_results_top=dir_results_top,
            dir_sub_results=dir_sub_results,
            dir_sub_normals_rgb=dir_sub_normals_rgb,
            dir_sub_normals_exr=dir_sub_normals_exr,
        )

        csv_writer.write_row(
            [
                config.eval.model,
                key,
                loss_mean,
                loss_median,
                percent_1,
                percent_2,
                percent_3,
            ]
        )

        print(f"\nmetrics - mean: {loss_mean:.4f} deg, median: {loss_median:.4f} deg, "+\
               f"P1: {percent_1:.4f}%, P2: {percent_2:.4f}%, P3: {percent_3:.4f}%\n\n")
    return

def main():
    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(
        description="Run eval of surface normal prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config_file", required=True, help="Path to config yaml file", metavar="path/to/config")
    ARGS = parser.parse_args()
    start_evaluation(ARGS)
    return


if __name__ == "__main__":
    main()
