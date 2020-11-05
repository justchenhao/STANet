"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import ntpath


def save_images(images, img_dir, name):
    """save images in img_dir, with name
    iamges: torch.float, B*C*H*W
    img_dir: str
    name: list [str]
    """
    for i, image in enumerate(images):
        print(image.shape)
        image_numpy = tensor2im(image.unsqueeze(0),normalize=False)*255
        basename = os.path.basename(name[i])
        print('name:', basename)
        save_path = os.path.join(img_dir,basename)
        save_image(image_numpy,save_path)


def save_visuals(visuals,img_dir,name):
    """
    """
    name = ntpath.basename(name)
    name = name.split(".")[0]
    print(name)
    # save images to the disk
    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        img_path = os.path.join(img_dir, '%s_%s.png' % (name, label))
        save_image(image_numpy, img_path)


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
