from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle


def andres_forward(x, a=10):
    return 2 * x / (x + a) - 1

def andres_backward(y, a=10):
    """Inverse of the function forward map.

    Numpy version
    """
    #simple_max = andres_forward(real_max, a)
    #simple_min = andres_forward(0, a)
    #y_clipped = np.clip(y, simple_min, simple_max)
    return a * (y + 1) / (1 - y)

def veldiv_backward(img, scale1=-700, scale2=400):
    img[img < 0] *= np.abs(scale1)
    img[img > 0] *= np.abs(scale2)
    return img

def tensor2im(input_image, imtype=np.float64, field_type='den'):
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
        image_numpy = andres_backward(image_numpy, a=7)  # A sufficiently (safe) high value is chosen for real_max.
        #return np.exp(image_numpy*10.077805519104004)
        #image_numpy = np.exp(image_numpy * 10)
        #return image_numpy.astype(imtype)
        #return inv_yeojohnson(image_numpy, lmbda=-2.8214043525721)
        #return (1/(image_numpy*9.93968))**2
        #image_numpy = custom_density_backward(image_numpy)
    elif field_type == 'veldiv':
        image_numpy = veldiv_backward(image_numpy, scale1=-900, scale2=400)  # image_numpy is 256X256 size.
        #return image_numpy.astype(imtype)
        #image_numpy = veldiv_backward_numpy(image_numpy, -13257.988, 12448.369)
    return image_numpy.astype(imtype)

#def tensor2im(input_image, imtype=np.uint8):
#    """"Convert a Tensor array into a numpy image array.
#    Parameters:
#        input_image (tensor) --  the input image tensor array
#        imtype (type)        --  the desired type of the converted numpy array
#    """
#    if not isinstance(input_image, np.ndarray):
#        if isinstance(input_image, torch.Tensor):  # get the data from a variable
#            image_tensor = input_image.data
#        else:
#            return input_image
#        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
#        if image_numpy.shape[0] == 1:  # grayscale to RGB
#            image_numpy = np.tile(image_numpy, (3, 1, 1))
#        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
#    else:  # if it is a numpy array, do nothing
#        image_numpy = input_image
#    return image_numpy.astype(imtype)


def tensor2vec(vector_tensor):
    numpy_vec = vector_tensor.data.cpu().numpy()
    if numpy_vec.ndim == 4:
        return numpy_vec[:, :, 0, 0]
    else:
        return numpy_vec


def pickle_load(file_name):
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_save(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def interp_z(z0, z1, num_frames, interp_mode='linear'):
    zs = []
    if interp_mode == 'linear':
        for n in range(num_frames):
            ratio = n / float(num_frames - 1)
            z_t = (1 - ratio) * z0 + ratio * z1
            zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    if interp_mode == 'slerp':
        z0_n = z0 / (np.linalg.norm(z0) + 1e-10)
        z1_n = z1 / (np.linalg.norm(z1) + 1e-10)
        omega = np.arccos(np.dot(z0_n, z1_n))
        sin_omega = np.sin(omega)
        if sin_omega < 1e-10 and sin_omega > -1e-10:
            zs = interp_z(z0, z1, num_frames, interp_mode='linear')
        else:
            for n in range(num_frames):
                ratio = n / float(num_frames - 1)
                z_t = np.sin((1 - ratio) * omega) / sin_omega * z0 + np.sin(ratio * omega) / sin_omega * z1
                zs.append(z_t[np.newaxis, :])
        zs = np.concatenate(zs, axis=0).astype(np.float32)

    return zs


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    np.save(image_path, image_numpy)

    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
    """

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
