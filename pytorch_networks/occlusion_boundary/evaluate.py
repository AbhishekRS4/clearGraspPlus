"""Run inference on outlines prediction model
"""

import os
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
from dataset import get_augumentation_list, get_data_loader, load_concat_sub_datasets


def evaluate(model, test_loader, device, num_classes, precision=5):
    model.eval()
    running_loss = 0.0
    running_iou = []
    running_tp = []
    running_tn = []
    running_fp = []
    running_fn = []
    num_images = len(test_loader.dataset)  # Num of total images

    with torch.no_grad():
        for ii, sample_batched in enumerate(tqdm(test_loader)):
            inputs, labels = sample_batched

            # Forward pass of the mini-batch
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
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

    mean_iou = round(sum(running_iou) / num_images, precision)

    return mean_iou


def start_evaluation(ARGS):
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
    SUBDIR_RESULT = "results"
    SUBDIR_OUTLINES = "outlines_files"
    checkpoint_epoch_num = config.eval.pathWeightsFile.split("/")[-1].split(".")[0].split("_")[-1]

    DIR_RESULTS_ROOT = config.eval.resultsDir
    DIR_RESULTS = os.path.join(DIR_RESULTS_ROOT, f"object_seg_epoch_{checkpoint_epoch_num}")
    if not os.path.isdir(DIR_RESULTS):
        os.makedirs(DIR_RESULTS)

    if config.eval.saveResultImages:
        try:
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_RESULT))
            os.makedirs(os.path.join(DIR_RESULTS, SUBDIR_MASKS))
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
    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(
        description="Run eval of outlines prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config_file", required=True, help="Path to yaml config file", metavar="path/to/config.yaml")
    ARGS = parser.parse_args()
    start_evaluation(ARGS)
    return


if __name__ == "__main__":
    main()
