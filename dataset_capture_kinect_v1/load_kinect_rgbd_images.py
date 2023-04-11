import pickle
import numpy as np

def load_images_from_pickle(file_pickle):
    """
    A generator function that yields a list of images from the given pickle file

    ---------
    Arguments
    ---------
    file_pickle: str
        full path to the pickle file with images

    ------
    Yields
    ------
        a generator for the images in the pickle file
    """
    with open(file_pickle, "rb") as file_handler:
        while True:
            try:
                yield pickle.load(file_handler)
            except EOFError:
                break
    return

def get_color_images(file_pickle_color):
    """
    Function to get a list of color images from the saved pickle file

    ---------
    Arguments
    ---------
    file_pickle_color: str
        full path to the pickle file with color images

    -------
    Returns
    -------
    color_images: A list of color images loaded from the pickle file as numpy arrays
    """
    color_images = list(load_images_from_pickle(file_pickle_color))
    return color_images

def get_depth_images(file_pickle_depth):
    """
    Function to get a list of depth images from the saved pickle file

    ---------
    Arguments
    ---------
    file_pickle_depth: str
        full path to the pickle file with depth images

    -------
    Returns
    -------
    depth_images: A list of depth images loaded from the pickle file as numpy arrays
    """
    depth_images = list(load_images_from_pickle(file_pickle_depth))
    return depth_images
