"""
Evaluation script for object segmentation task
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
            iou, tp, tn, fp, fn = utils.compute_metrics(
                torch.squeeze(pred_labels).detach().cpu(), torch.squeeze(labels).detach().cpu()
            )
            running_iou.append(iou)
            running_tp.append(tp)
            running_tn.append(tn)
            running_fp.append(fp)
            running_fn.append(fn)

            """
            # Save Results
            # grid image with input, prediction and label
            pred_rgb = utils.label_to_rgb(torch.unsqueeze(pred, 0))
            label_rgb = utils.label_to_rgb(label)

            images = torch.cat((img, pred_rgb, label_rgb), dim=2)
            grid_image = make_grid(images, 1, normalize=True, scale_each=True)
            numpy_grid = grid_image * 255  # Scale from range [0.0, 1.0] to [0, 255]
            numpy_grid = numpy_grid.numpy().transpose(1, 2, 0).astype(np.uint8)
            imageio.imwrite(result_path, numpy_grid)

            result_mask_path = os.path.join(DIR_RESULTS, SUBDIR_MASKS,
                                            "{:09d}-mask.png".format(ii * config.eval.batchSize + iii))
            imageio.imwrite(result_mask_path, (pred.squeeze(0).numpy() * 255).astype(np.uint8))
            """

    mean_iou = round(sum(running_iou) / num_images, precision)
    mean_tp = round(sum(running_tp) / num_images, precision)
    mean_tn = round(sum(running_tn) / num_images, precision)
    mean_fp = round(sum(running_fp) / num_images, precision)
    mean_fn = round(sum(running_fn) / num_images, precision)
    return mean_iou, mean_tp, mean_tn, mean_fp, mean_fn


def start_evaluation(ARGS):
    ###################### Load Config File #############################
    FILE_PATH_CONFIG = ARGS.config_file
    with open(FILE_PATH_CONFIG) as fd_config_yaml:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print("Inference of Object Segmentation model. Loading checkpoint...")

    ###################### Load Checkpoint and its data #############################
    if not os.path.isfile(config.eval.pathWeightsFile):
        raise ValueError(f"Invalid path to the given weights file in config. The file {config.eval.pathWeightsFile} does not exist")

    CHECKPOINT = torch.load(config.eval.pathWeightsFile, map_location="cpu")
    if "model_state_dict" in CHECKPOINT:
        print(f"Loaded data from checkpoint weights file: {config.eval.pathWeightsFile}")
    else:
        raise ValueError("The checkpoint file does not have model_state_dict in it.\
                         Please use the newer checkpoint files!")

    # Create directory to save results
    SUBDIR_RESULT = "results"
    SUBDIR_MASKS = "masks"
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
    logging_column_names = ["model_name", "dataset", "mean_IoU", "TP", "TN", "FP", "FN"]
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
        dict_dataset_loader.update({'real': test_real_loader})
    if db_test_syn:
        dict_dataset_loader.update({'synthetic': test_syn_loader})

    ### Run evaluation on the Test Set ###
    print("\nInference - transparent object segmentation task")
    print("=" * 50 + "\n")

    for key in dict_dataset_loader:
        print("\n" + key + ":")
        print("=" * 30)

        test_loader_current = dict_dataset_loader[key]
        num_images_current = len(test_loader_current.dataset)

        if num_images_current == 0:
            continue

        mean_iou, mean_tp, mean_tn, mean_fp, mean_fn = evaluate(
            model,
            test_loader_current,
            device,
            config.eval.numClasses,
        )

        print(f"test set: {key}, num images: {num_images_current}")
        print(f"\nevaluation metrics, mean IoU: {mean_iou:.4f}, TP: {mean_tp:.4f} %"+\
              f", TN: {mean_tn:.4f} %, FP: {mean_fp:.4f} %, FN: {mean_fn:.4f} %")
        csv_writer.write_row(
            [
                config.eval.model,
                key,
                mean_iou,
                mean_tp,
                mean_tn,
                mean_fp,
                mean_fn,
            ]
        )
    return

def main():
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
