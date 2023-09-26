import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_samples_per_obj_distribution(ARGS):
    # for train / validation sets
    train_objects = [
        "cup-with-waves",
        "flower-bath-bomb",
        "heart-bath-bomb",
        "square-plastic-bottle",
        "stemless-plastic-champagne-glass"
    ]
    list_counts_train = [4925, 5270, 11662, 11702, 11895]
    list_counts_valid = [100, 100, 100, 115, 117]
    x_axis_train = np.arange(len(train_objects))

    fig = plt.figure(1)
    plt.bar(x_axis_train - 0.2, list_counts_train, 0.4, label = "train set")
    plt.bar(x_axis_train + 0.2, list_counts_valid, 0.4, label = "validation set")
    plt.yticks(fontsize=ARGS.font_size)
    plt.xticks(x_axis_train, train_objects, fontsize=ARGS.font_size)
    plt.xlabel("Objects", fontsize=ARGS.font_size)
    plt.ylabel("Number of samples", fontsize=ARGS.font_size)
    plt.title("Distribution of samples for each object in train/validation sets", fontsize=ARGS.font_size)
    plt.legend(prop={"size": ARGS.font_size})
    plt.grid()
    plt.show()

    # for test set
    test_objects = [
        "glass-round-potion",
        "glass-square-potion",
        "star-bath-bomb",
        "tree-bath-bomb"
    ]
    list_counts_test = [109, 94, 98, 107]
    x_axis_test = np.arange(len(test_objects))

    fig = plt.figure(2)
    plt.bar(x_axis_test, list_counts_test, 0.4, label = "test set")
    plt.yticks(fontsize=ARGS.font_size)
    plt.xticks(x_axis_test, test_objects, fontsize=ARGS.font_size)
    plt.xlabel("Objects", fontsize=ARGS.font_size)
    plt.ylabel("Number of samples", fontsize=ARGS.font_size)
    plt.title("Distribution of samples for each object in test set", fontsize=ARGS.font_size)
    plt.legend(prop={"size": ARGS.font_size})
    plt.grid()
    plt.show()
    return


def main():
    font_size = 25

    parser = argparse.ArgumentParser(
        description="Script to plot number of samples per object distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--font_size", default=font_size, type=int,
        help="font size of the text in the matplotlib plots")

    ARGS = parser.parse_args()
    plot_samples_per_obj_distribution(ARGS)
    return

main()
