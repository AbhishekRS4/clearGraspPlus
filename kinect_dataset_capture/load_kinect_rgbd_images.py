import pickle
import numpy as np

def load_images_from_pickle(file_pickle):
    with open(file_pickle, "rb") as file_handler:
        while True:
            try:
                yield pickle.load(file_handler)
            except EOFError:
                break
    return

def get_color_images(file_pickle_color):
    color_images = list(load_images_from_pickle(file_pickle_color))
    return color_images

def get_depth_images(file_pickle_depth):
    depth_images = list(load_images_from_pickle(file_pickle_depth))
    return depth_images
