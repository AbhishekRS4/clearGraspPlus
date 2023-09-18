import os
import sys
import torch
import argparse
import numpy as np

from imageio import imread, imwrite

from utils import utils

def save_label_visualization(ARGS):
    if os.path.isfile(ARGS.input_label_file):
        img_label = utils.exr_loader(ARGS.input_label_file, ndim=3)

        # convert numpy array to torch tensor
        img_label = torch.from_numpy(img_label)

        # convert label to visualization mask
        img_vis_label = utils.normal_to_rgb(img_label)
        # print(img_vis_label.shape)

        # convert tensor to numpy array and from [0.0, 1.0] to [0, 255]
        img_vis_label = img_vis_label.detach().cpu().squeeze().numpy() * 255

        # convert [3, H, W] to [H, W, 3]
        img_vis_label = np.transpose(img_vis_label, [1, 2, 0]).astype(np.uint8)
        # print(img_vis_label.shape)

        imwrite(ARGS.output_vis_file, img_vis_label)
        print(f"output visualization successfully saved to [{ARGS.output_vis_file}]")
    else:
        print(f"the file [{ARGS.input_label_file}] does not exist")
    return

def main():
    parser = argparse.ArgumentParser(
        description="save visualization of the surface normal label",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_label_file", type=str,
        required=True, help="full path to image label file to load")
    parser.add_argument("--output_vis_file", type=str,
        required=True, help="full path to image visualization file to save")
    ARGS = parser.parse_args()
    save_label_visualization(ARGS)
    return

if __name__ == "__main__":
    main()
