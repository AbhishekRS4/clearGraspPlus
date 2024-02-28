#!/usr/bin/env python3
'''Script to run Depth Completion on Synthetic and Real datasets, visualizing the results and computing the error metrics.
This will save all intermediate outputs like surface normals, etc, create a collage of all the inputs and outputs and
create pointclouds from the input, modified input and output depth images.
'''
import argparse
import csv
import glob
import os
import shutil
import sys
import itertools

# Importing Pytorch before Open3D can cause unknown "invalid pointer" error
import open3d as o3d
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import depth_completion_api
from api import utils as api_utils

import attrdict
import imageio
import termcolor
import yaml
import torch
import cv2
import numpy as np

def main(args):
    # Load Config File
    CONFIG_FILE_PATH = args.configFile
    with open(CONFIG_FILE_PATH) as fd:
        config_yaml = yaml.safe_load(fd)
    config = attrdict.AttrDict(config_yaml)

    # Create directory to save results
    results_dir = args.dir_results
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    shutil.copy2(CONFIG_FILE_PATH, os.path.join(results_dir, 'config.yaml'))
    print('\nSaving results to folder: ' + termcolor.colored('"{}"\n'.format(results_dir), 'green'))

    # Init depth completion API
    outputImgHeight = int(config.depth2depth.yres)
    outputImgWidth = int(config.depth2depth.xres)

    depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=config.normals.pathWeightsFile,
                                                                outlinesWeightsFile=config.outlines.pathWeightsFile,
                                                                masksWeightsFile=config.masks.pathWeightsFile,
                                                                normalsModel=config.normals.model,
                                                                outlinesModel=config.outlines.model,
                                                                masksModel=config.masks.model,
                                                                depth2depthExecutable=config.depth2depth.pathExecutable,
                                                                outputImgHeight=outputImgHeight,
                                                                outputImgWidth=outputImgWidth,
                                                                fx=int(config.depth2depth.fx),
                                                                fy=int(config.depth2depth.fy),
                                                                cx=int(config.depth2depth.cx),
                                                                cy=int(config.depth2depth.cy),
                                                                filter_d=config.outputDepthFilter.d,
                                                                filter_sigmaColor=config.outputDepthFilter.sigmaColor,
                                                                filter_sigmaSpace=config.outputDepthFilter.sigmaSpace,
                                                                maskinferenceHeight=config.masks.inferenceHeight,
                                                                maskinferenceWidth=config.masks.inferenceWidth,
                                                                normalsInferenceHeight=config.normals.inferenceHeight,
                                                                normalsInferenceWidth=config.normals.inferenceWidth,
                                                                outlinesInferenceHeight=config.normals.inferenceHeight,
                                                                outlinesInferenceWidth=config.normals.inferenceWidth,
                                                                min_depth=config.depthVisualization.minDepth,
                                                                max_depth=config.depthVisualization.maxDepth,
                                                                tmp_dir=results_dir)

    # Create lists of input data
    rgb_file_list = []
    depth_file_list = []

    rgb_file_list = sorted(os.listdir(os.path.join(args.dir_data, "color_images")))
    depth_file_list = sorted(os.listdir(os.path.join(args.dir_data, "depth_images")))

    print('Total Num of rgb_files:', len(rgb_file_list))
    print('Total Num of depth_files:', len(depth_file_list))
    assert len(rgb_file_list) > 0, ('No files found in given directories')

    for i in range(len(rgb_file_list)):
        # Run Depth Completion
        color_img = imageio.imread(rgb_file_list[i])
        input_depth = imageio.imread(depth_file_list[i])

        try:
            output_depth, filtered_output_depth = depthcomplete.depth_completion(
                color_img,
                input_depth,
                inertia_weight=float(config.depth2depth.inertia_weight),
                smoothness_weight=float(config.depth2depth.smoothness_weight),
                tangent_weight=float(config.depth2depth.tangent_weight),
                mode_modify_input_depth=config.modifyInputDepth.mode,
                dilate_mask=True)
        except depth_completion_api.DepthCompletionError as e:
            print('Depth Completion Failed:\n  {}\n  ...skipping image {}'.format(e, i))
            continue

        # Save Results of Depth Completion
        error_output_depth, error_filtered_output_depth = depthcomplete.store_depth_completion_outputs(
            root_dir=results_dir,
            files_prefix=i,
            min_depth=config.depthVisualization.minDepth,
            max_depth=config.depthVisualization.maxDepth)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval of depth completion on synthetic data')
    parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
    parser.add_argument('-m', '--maskInputDepth', action="store_true", help='Whether we should mask out objects in input depth')
    parser.add_argument("-d", "--dir_data", required=True, help="full path to directory containing the data")
    parser.add_argument("-r", "--dir_results", required=True, help="full path to directory where results need to be saved")
    args = parser.parse_args()
    main(args)
