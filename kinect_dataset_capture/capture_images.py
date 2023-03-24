import os
import cv2
import time
import pickle
import signal
import datetime
import argparse
import numpy as np

import freenect
import frame_convert2

is_capture = True

def exit_handler(sig_num, stack_frame):
    global is_capture
    is_capture = False

def get_depth_frame(kinect_device_index):
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth(kinect_device_index)[0])

def get_color_frame(kinect_device_index):
    return freenect.sync_get_video(kinect_device_index)[0]

def start_image_capture(ARGS):
    # define exit handlers
    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)

    print("Saving colour and depth frames")

    # define file names
    file_name_depth = os.path.join(ARGS.dir_dataset, "depth_images.pickle")
    file_name_color = os.path.join(ARGS.dir_dataset, "color_images.pickle")

    # open file descriptors for color and depth pickle files
    fd_depth = open(file_name_depth, "wb")
    fd_color = open(file_name_color, "wb")

    # initialize some variables
    kinect_device_index = 0
    frame_count = 0

    # Save color and depth frames
    # Actual recording loop, exit by pressing ctrl+c
    while is_capture:
        try:
            # get both color and depth frames
            depth_frame = get_depth_frame(kinect_device_index)
            color_frame = get_color_frame(kinect_device_index)

            print("depth frame shape: ", depth_frame.shape)
            print("color frame shape: ", color_frame.shape)

            # dump the color and depth frames into pickle files
            pickle.dump(depth_frame, fd_depth)
            pickle.dump(color_frame, fd_color)

            # increment the frame count
            frame_count += 1
        except TypeError:
            kinect_device_index = 0
            continue
    print(f"saved {frame_count} frames")

    # close the pickle file descriptors
    print("closing the file handlers")
    fd_depth.close()
    fd_color.close()
    print("successfully closed the file handlers")
    return

def main():
    dir_dataset = os.getcwd()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="full directory path where the dataset needs to be saved")

    ARGS, unparsed = parser.parse_known_args()
    start_image_capture(ARGS)
    return

if __name__ == "__main__":
    main()
