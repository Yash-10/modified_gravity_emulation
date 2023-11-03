"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import pandas as pd
import os
import torch
from data.base_dataset import andres_forward, veldiv_forward

def andres_backward(y, shift=0., scale=1., real_max=1e8):
    """Inverse of the function forward map.
    Numpy version
    """
    y = y.astype(np.float64)
    simple_max = andres_forward(real_max, shift, scale)
    simple_min = andres_forward(0, shift, scale)
    y_clipped = np.clip(y, simple_min, simple_max) / scale
    return (shift + 1) * (y_clipped + 1) / (1 - y_clipped)

def custom_density_backward(arr, a, b, eps=1e-5):
    return np.exp((arr * (b-a+eps)+b+a)/2)

#def veldiv_backward(img, minval, maxval):
#    scaled = (img + 1) / 2
#    scaled = scaled.float()
#    arr_range = maxval - minval
#    arr = scaled * float(arr_range) + minval
#    return arr

#def veldiv_backward_numpy(img, minval, maxval):
#    scaled = (img + 1) / 2
#    scaled = scaled.astype('f')
#    arr_range = maxval - minval
#    arr = scaled * float(arr_range) + minval
#    return arr

def veldiv_backward(img, scale=400):
    img = img * scale
    return img

#def veldiv_backward(img):
#    print(img.min(), img.max())
#    return img * 13257.988

# Note this pixel applies to pixel values individiually
#def invert_yeojhonson(value, lmbda):
#  if value>= 0 and lmbda == 0:
#    return np.exp(value) - 1
#  elif value >= 0 and lmbda != 0:
#    return (value * lmbda + 1) ** (1 / lmbda) - 1
#  elif value < 0 and lmbda != 2:
#    return 1 - (-(2 - lmbda) * value + 1) ** (1 / (2 - lmbda))
#  elif value < 0 and lmbda == 2:
#    return 1 - np.exp(-value)

def inv_yeojohnson(x, lmbda):
    x = x.astype(np.complex64)
    x_inv = np.zeros_like(x)
    pos = x >= 0

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.exp(x[pos]) - 1
    else:  # lmbda != 0
        x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
    else:  # lmbda == 2
        x_inv[~pos] = 1 - np.exp(-x[~pos])

    return np.real(x_inv)

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
        #image_numpy = andres_backward(image_numpy, scale=1., shift=3, real_max=1e8)  # A sufficiently (safe) high value is chosen for real_max.
        #return np.exp(image_numpy*10.077805519104004)
        #image_numpy = np.exp(image_numpy * 10)
        return image_numpy
        #return inv_yeojohnson(image_numpy, lmbda=-2.8214043525721)
        #return (1/(image_numpy*9.93968))**2
        #image_numpy = custom_density_backward(image_numpy)
    elif field_type == 'veldiv':
        #image_numpy = veldiv_backward(image_numpy, scale=1230.653717041015)  # image_numpy is 512X512 size.
        return image_numpy
        #image_numpy = veldiv_backward_numpy(image_numpy, -13257.988, 12448.369)
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
