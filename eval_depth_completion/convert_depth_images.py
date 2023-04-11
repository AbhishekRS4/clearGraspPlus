import os
import sys
import imageio
import argparse

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api import utils as api_utils

def convert_depth_images(ARGS):
    EXT_DEPTH_IMG = "-transparent-depth-img.exr"
    list_depth_files = [f for f in os.listdir(ARGS.dir_depth_exr_data) if f.endswith(EXT_DEPTH_IMG)]
    num_depth_files = len(list_depth_files)

    print(f"Num depth files to convert: {num_depth_files}")

    if not os.path.isdir(ARGS.dir_out_depth_png_data):
        os.makedirs(ARGS.dir_out_depth_png_data)

    for file_index in range(num_depth_files):
        file_depth = os.path.join(ARGS.dir_depth_exr_data, list_depth_files[file_index])
        input_depth = api_utils.exr_loader(file_depth, ndim=1)

        print(input_depth.shape, np.min(input_depth), np.max(input_depth))
        #imageio.imsave()

    return

def main():
    parser = argparse.ArgumentParser(description="convert format of depth images")
    parser.add_argument("--dir_depth_exr_data", required=True,
        help="full directory path containing depth exr files")
    parser.add_argument("--dir_out_depth_png_data", required=True,
        help="full directory path where output depth images need to be saved")
    args = parser.parse_args()
    convert_depth_images(args)
    return

if __name__ == "__main__":
    main()
