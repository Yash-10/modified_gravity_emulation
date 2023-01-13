"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os
import torch
from data.base_dataset import andres_forward, veldiv_forward

def andres_backward(y, shift=20., scale=1., real_max=1e8):
    """Inverse of the function forward map.
    Numpy version
    """
    simple_max = andres_forward(real_max, shift, scale)
    simple_min = andres_forward(0, shift, scale)
    y_clipped = np.clip(y, simple_min, simple_max) / scale
    return (shift + 1) * (y_clipped + 1) / (1 - y_clipped)

def veldiv_backward(img, minval, maxval):
    scaled = (img + 1) / 2
    scaled = scaled.float()
    arr_range = maxval - minval
    arr = scaled * float(arr_range) + minval
    return arr

def veldiv_backward_numpy(img, minval, maxval):
    scaled = (img + 1) / 2
    scaled = scaled.astype('f')
    arr_range = maxval - minval
    arr = scaled * float(arr_range) + minval
    return arr

def tensor2im(input_image, imtype=np.float32, field_type='den'):
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
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    
    if field_type == 'den':
        # scale and shift values must match those used in `andres_forward`.
        image_numpy = andres_backward(image_numpy, scale=1., shift=1., real_max=1.5e4)  # In the original images, max value is always less than 1.5e4.
    else:
        image_numpy = veldiv_backward_numpy(image_numpy, -13257.988, 12448.369)
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    np.save(image_path, image_numpy)


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
