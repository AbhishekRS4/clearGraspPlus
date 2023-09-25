import os
import argparse
import numpy as np

from imageio import imread

from utils import utils

def convert_nyu_sur_nor_rgb_to_exr(ARGS):
    list_nyu_sur_nor_rgb_files = os.listdir(ARGS.dir_input_rgb)

    print("converting NYU surface normal RGB labels to EXR labels")
    print(f"loading NYU surface normal RGB label files from: {ARGS.dir_input_rgb}")
    print(f"number of label files: {len(list_nyu_sur_nor_rgb_files)}")

    if not os.path.isdir(ARGS.dir_output_exr):
        os.makedirs(ARGS.dir_output_exr)

    for file_nyu_sur_nor_rgb in list_nyu_sur_nor_rgb_files:
        # load sur nor label in RGB format
        img_nyu_sn_rgb = imread(os.path.join(ARGS.dir_input_rgb, file_nyu_sur_nor_rgb))

        # set output file name
        file_nyu_exr = os.path.join(ARGS.dir_output_exr, file_nyu_sur_nor_rgb.split(".")[0]+"-cameraNormals.exr")

        # convert from [0, 255] to [-1., 1.]
        img_nyu_sn_normalized = (2. * img_nyu_sn_rgb / 255.) - 1.
        img_nyu_sn_normalized = img_nyu_sn_normalized.astype(np.float32)

        # save sur nor label in EXR format
        utils.exr_saver(file_nyu_exr, np.transpose(img_nyu_sn_normalized, axes=[2, 0, 1]))
    print(f"saved NYU surface normal EXR label files in: {ARGS.dir_output_exr}")
    return


def main():
    parser = argparse.ArgumentParser(
        description="convert NYU surface normals labels from RGB to EXR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir_input_rgb", type=str,
        required=True, help="full path to directory with surface normal RGB labels")
    parser.add_argument("--dir_output_exr", type=str,
        required=True, help="full path to directory to save surface normal labels in EXR")
    ARGS = parser.parse_args()
    convert_nyu_sur_nor_rgb_to_exr(ARGS)
    return

if __name__ == "__main__":
    main()
