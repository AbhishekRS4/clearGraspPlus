"""
Evaluation script for occlusion boundary prediction task
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
from dataset import get_augumentation_list, get_data_loader, load_concat_sub_datasets


def evaluate(model, test_loader, device, num_classes,
    dir_results_top=None, dir_sub_occ_weights_rgb=None,
    dir_sub_occ_boundaries=None, dir_sub_overlay=None, dir_sub_results=None,
    precision=5, eps=1):

    model.eval()
    running_loss = 0.0
    running_iou = []
    running_tp = []
    running_tn = []
    running_fp = []
    running_fn = []
    num_images = len(test_loader.dataset)  # Num of total images

    with torch.no_grad():
        for ii, batch in enumerate(tqdm(test_loader)):
            inputs, labels, image_path = batch

            # Forward pass of the mini-batch
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            output_softmax = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()
            pred_labels = torch.argmax(outputs, 1)

            #print("pred:", pred_labels.shape, pred_labels.dtype, pred_labels.min(), pred_labels.max())
            #print("label:", labels.shape, labels.dtype, labels.min(), labels.max())
            """
            _total_iou, _, _ = utils.get_iou(pred_labels.detach().cpu().numpy(),
                                             labels.long().squeeze(1).detach().cpu().numpy(),
                                             n_classes=num_classes)
            """
            _total_iou = utils.compute_mean_IOU(labels, pred_labels, num_classes=num_classes)

            running_iou.append(_total_iou)

            if dir_results_top is not None:
                dataset_string = image_path[0].split("/")[-3]
                image_string = image_path[0].split("/")[-1].split(".")[0]

                # save occlusion weights
                occ_weights = 1 - output_softmax
                final_occ_weights = np.power(occ_weights, 3)
                final_occ_weights = np.multiply(final_occ_weights, 1000).astype(np.int16)

                final_occ_weights[final_occ_weights == 0] += eps
                final_occ_weights[final_occ_weights == 1000] -= eps
                final_occ_weights = np.transpose(np.squeeze(final_occ_weights), (1, 2, 0))
                #print(final_occ_weights.shape)
                #sys.exit(0)

                # save the occlusion weights' RGB visualization
                final_occ_weights_rgb = (occ_weights * 255).astype(np.uint8)
                final_occ_weights_rgb = np.transpose(np.squeeze(final_occ_weights_rgb), (1, 2, 0))
                #print(final_occ_weights_rgb.shape)
                #sys.exit(0)
                final_occ_weights_rgb = cv2.applyColorMap(final_occ_weights_rgb, cv2.COLORMAP_OCEAN)
                final_occ_weights_rgb = cv2.cvtColor(final_occ_weights_rgb, cv2.COLOR_BGR2RGB)
                file_path_occ_weights_rgb = os.path.join(
                    dir_results_top, dir_sub_occ_weights_rgb,
                    f"{dataset_string}_{image_string}_occ_weights_rgb.png"
                )
                imageio.imwrite(file_path_occ_weights_rgb, final_occ_weights_rgb)

                # save RGB viz of occlusion boundaries
                input_image = np.squeeze(inputs.detach().cpu().numpy())
                #print(input_image.shape)
                input_image = (input_image.transpose(1, 2, 0) * 255).astype(np.uint8)

                pred_labels = np.squeeze(pred_labels.detach().cpu().numpy())

                result_rgb_img = np.zeros_like(input_image)
                #print(result_rgb_img.shape, pred_labels.shape)
                result_rgb_img[pred_labels == 0, 0] = 255 # R
                result_rgb_img[pred_labels == 1, 1] = 255 # G
                result_rgb_img[pred_labels == 2, 2] = 255 # B
                #result_rgb_img = cv2.resize(result_rgb_img, (512, 288), interpolation=cv2.INTER_NEAREST)
                file_path_result_rgb = os.path.join(
                    dir_results_top, dir_sub_occ_boundaries,
                    f"{dataset_string}_{image_string}_occ_boundary.png"
                )
                imageio.imwrite(file_path_result_rgb, result_rgb_img)

                # save overlay of result on RGB image.
                mask_rgb = input_image.copy()

                mask_rgb[pred_labels == 1, 1] = 255
                mask_rgb[pred_labels == 2, 0] = 255
                overlay_img = cv2.addWeighted(mask_rgb, 0.6, input_image, 0.4, 0)
                file_path_overlay = os.path.join(
                    dir_results_top, dir_sub_overlay,
                    f"{dataset_string}_{image_string}_overlay.png"
                )
                imageio.imwrite(file_path_overlay, overlay_img)

                # save resulting grid image with input, prediction and label
                # print(pred_labels.shape)
                # print(labels.shape)
                output_prediction_rgb = utils.label_to_rgb(pred_labels)
                label_rgb = utils.label_to_rgb(labels)

                # print((inputs.detach().cpu().shape, output_prediction_rgb.shape, label_rgb.shape))
                grid_image = torch.cat(
                    (
                        inputs.detach().cpu().squeeze(),
                        output_prediction_rgb, label_rgb.squeeze()
                    ),
                    dim=2
                )
                grid_image = make_grid(grid_image, 1, normalize=True, scale_each=True)
                numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
                numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)
                file_path_occ_boun_result = os.path.join(
                    dir_results_top, dir_sub_results,
                    f"{dataset_string}_{image_string}_result.png"
                )
                imageio.imwrite(file_path_occ_boun_result, numpy_grid)


    mean_iou = round(sum(running_iou) / num_images, precision)

    return mean_iou


def start_evaluation(ARGS):
    ###################### Load Config File #############################
    FILE_PATH_CONFIG = ARGS.config_file
    with open(FILE_PATH_CONFIG) as fd_config_yaml:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print("Inference of occlusion boundary prediction model. Loading checkpoint...")

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
    SUBDIR_OCC_WEIGHTS_RGB = "occ_weights_rgb"
    SUBDIR_OCC_BOUNDARIES = "occ_boundaries"
    SUBDIR_OVERLAY = "overlay"
    SUBDIR_RESULT = "results"

    checkpoint_epoch_num = config.eval.pathWeightsFile.split("/")[-1].split(".")[0].split("_")[-1]

    DIR_RESULTS_ROOT = config.eval.resultsDir
    DIR_RESULTS = os.path.join(DIR_RESULTS_ROOT, f"occ_bound_epoch_{checkpoint_epoch_num}")
    if not os.path.isdir(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)

    if config.eval.saveResultImages:
        try:
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_OCC_WEIGHTS_RGB))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_OCC_BOUNDARIES))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_OVERLAY))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_RESULT))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

    print(f"Saving results to folder: {DIR_RESULTS}")
    # Create CSV File to store error metrics
    FILE_NAME_LOGS_CSV = f"computed_metrics_epoch_{checkpoint_epoch_num}.csv"
    logging_column_names = ["model_name", "dataset", "mean_IoU"]
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

    if (db_test_syn is not None) and (db_test_real is not None):
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
    if config.eval.model == "deeplab_xception":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="xception", sync_bn=True, freeze_bn=False)
    elif config.eval.model == "deeplab_resnet":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="resnet", sync_bn=True, freeze_bn=False)
    elif config.eval.model == "drn":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="drn", sync_bn=True, freeze_bn=False)
    elif config.eval.model == "drn_psa":
        model = deeplab.DeepLab(num_classes=config.eval.numClasses, backbone="drn_psa", sync_bn=True,
                                freeze_bn=False)  # output stride is 8 for drn_psa
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
    print("\nInference - occlusion boundary prediction task")
    print("=" * 50 + "\n")

    dir_results_top = None
    dir_sub_occ_weights_rgb = None
    dir_sub_occ_boundaries = None
    dir_sub_overlay = None
    dir_sub_results = None
    if config.eval.saveResultImages:
        dir_results_top = DIR_RESULTS
        dir_sub_occ_weights_rgb = SUBDIR_OCC_WEIGHTS_RGB
        dir_sub_occ_boundaries = SUBDIR_OCC_BOUNDARIES
        dir_sub_overlay = SUBDIR_OVERLAY
        dir_sub_results = SUBDIR_RESULT

    for key in dict_dataset_loader:
        print("\n" + key + ":")
        print("=" * 30)

        test_loader_current = dict_dataset_loader[key]
        num_images_current = len(test_loader_current.dataset)

        if num_images_current == 0:
            continue

        print(f"test set: {key}, num images: {num_images_current}")
        mean_iou = evaluate(
            model,
            test_loader_current,
            device,
            config.eval.numClasses,
            dir_results_top=dir_results_top,
            dir_sub_occ_weights_rgb=dir_sub_occ_weights_rgb,
            dir_sub_occ_boundaries=dir_sub_occ_boundaries,
            dir_sub_overlay=dir_sub_overlay,
            dir_sub_results=dir_sub_results,
        )

        print(f"\nevaluation metrics, mean IoU: {mean_iou:.4f}")
        csv_writer.write_row(
            [
                config.eval.model,
                key,
                mean_iou,
            ]
        )
    return


def main():
    parser = argparse.ArgumentParser(
        description="Run eval of occlusion boundary prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config_file", required=True, help="Path to yaml config file", metavar="path/to/config.yaml")
    ARGS = parser.parse_args()
    start_evaluation(ARGS)
    return


if __name__ == "__main__":
    main()
