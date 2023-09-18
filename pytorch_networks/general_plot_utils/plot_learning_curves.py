import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curves(ARGS):
    # load logs and concat dataframes for the all the experiments in that task
    print(f"Loading data from {ARGS.dir_tasks}")
    print(f"Model: {ARGS.which_model}")
    print(f"Task: {ARGS.which_task}")
    list_exps_dirs = sorted(os.listdir(os.path.join(ARGS.dir_tasks, ARGS.which_task)))
    df_exps = None

    for dir_exp in list_exps_dirs:
        file_csv = os.path.join(ARGS.dir_tasks, ARGS.which_task, dir_exp, "train_logs.csv")
        df_exp = pd.read_csv(file_csv)
        if df_exps is None:
            df_exps = df_exp
        else:
            df_exps = pd.concat([df_exps, df_exp])
    print(df_exps.shape)

    # First, plot the losses
    fig = plt.figure(1)
    plt.title(f"Learning curve for {ARGS.which_task} task with {ARGS.which_model} model", fontsize=ARGS.font_size)
    plt.grid()
    if ARGS.which_task == "surface_normals":
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_loss_mean, label="train loss mean")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_loss_mean, label="validation loss mean")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_loss_median, label="train loss median")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_loss_median, label="validation loss median")
    else:
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_loss, label="train loss")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_loss, label="validation loss")
    plt.xlabel("epoch", fontsize=ARGS.font_size)
    plt.ylabel("loss", fontsize=ARGS.font_size)
    plt.xticks(fontsize=ARGS.font_size)
    plt.yticks(fontsize=ARGS.font_size)
    plt.legend(prop={"size": ARGS.font_size})
    plt.show()

    # Second, plot the metrics
    metric = None
    if ARGS.which_task == "surface_normals":
        metric = "thresholded % pixels"
    else:
        metric = "mean IoU score"

    fig = plt.figure(2)
    plt.title(f"Performance curve for {ARGS.which_task} task with {ARGS.which_model} model", fontsize=ARGS.font_size)
    plt.grid()
    if ARGS.which_task == "surface_normals":
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_pc_1, label=r"train thresholded @ $11.25 \degree$ deg")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_pc_2, label=r"train thresholded @ $22.5 \degree$ deg")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_pc_3, label=r"train thresholded @ $30 \degree$ deg")

        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_pc_1, label=r"validation thresholded @ $11.25 \degree$ deg")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_pc_2, label=r"validation thresholded @ $22.5 \degree$ deg")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_pc_3, label=r"validation thresholded @ $30 \degree$ deg")
    else:
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.train_iou, label="train mean IoU")
        plt.plot(np.arange(1, df_exps.shape[0] + 1), df_exps.valid_iou, label="validation mean IoU")
    plt.xlabel("epoch", fontsize=ARGS.font_size)
    plt.ylabel(metric, fontsize=ARGS.font_size)
    plt.xticks(fontsize=ARGS.font_size)
    plt.yticks(fontsize=ARGS.font_size)
    plt.legend(prop={"size": ARGS.font_size})
    plt.show()
    return


def main():
    dir_tasks = "/media/abhishek/Extreme SSD/ai_master_thesis/drn_psa_models"
    which_task = "object_segmentation"
    which_model = "ResNet50_PSA"
    font_size = 25

    parser = argparse.ArgumentParser(
        description="Script to plot training/validation losses/metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dir_tasks", default=dir_tasks, type=str,
        help="Path to config yaml file")
    parser.add_argument("--which_task", default=which_task, type=str,
        choices=["object_segmentation", "occlusion_boundary", "surface_normals"],
        help="Task for which the plots need to be generated")
    parser.add_argument("--which_model", default=which_model, type=str,
        choices=["DRN", "ResNet34_PSA", "ResNet50_PSA"],
        help="model name which needs to be used in the title")
    parser.add_argument("--font_size", default=font_size, type=int,
        help="font size of the text in the matplotlib plots")

    ARGS = parser.parse_args()
    plot_learning_curves(ARGS)
    return

main()
