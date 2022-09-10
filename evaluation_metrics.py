# Evaluation functions

import gc
from functools import partial
import numpy as np

# Plotting.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import Pk_library.Pk_library as PKL

from scipy.stats import wasserstein_distance
import scipy.ndimage.filters as filters

import torch
from torchmetrics.functional import multiscale_structural_similarity_index_measure

# Power spectrum
def ps_2d(delta, BoxSize=128):
    """Calculates the 2D power spectrum of a density field.

    Args:
        delta (numpy.ndarray): Density slice.
        BoxSize (float): Simulation box size.

    Returns:
        (numpy.ndarray, numpy.ndarray): The wavenumbers and power spectrum amplitudes.
    """
    MAS = 'None'
    threads = 2
    Pk2D2 = PKL.Pk_plane(delta, BoxSize, MAS, threads)
    # get the attributes of the routine
    k2      = Pk2D2.k      #k in h/Mpc
    Pk2     = Pk2D2.Pk     #Pk in (Mpc/h)^2
#     Nmodes = Pk2D2.Nmodes #Number of modes in the different k bins
    return k2, Pk2

# Wasserstein distance.
# Code taken from https://renkulab.io/gitlab/nathanael.perraudin/darkmattergan/-/blob/master/cosmotools/metric/evaluation.py
def wasserstein_distance_norm(p, q):
    """Computes 1-Wasserstein distance between standardized p and q arrays.
    Notes
    -----
    - p denotes real images and q denotes fake (or generated) images.
    - p and q both are of shape (n_examples, height, width).
    - p and q are standardized using mean and standard deviation of p.

    Args:
        p (numpy.ndarray): Real images.
        q (numpy.ndarray): Fake images.

    Returns:
        float: 1-Wasserstein distance between two sets of images.
    """
    mu, sig = p.mean(), p.std()
    p_norm = (p.flatten() - mu)/sig        
    q_norm = (q.flatten() - mu)/sig        
    return wasserstein_distance(p_norm, q_norm)

# Peak count.
# Code taken from https://renkulab.io/gitlab/nathanael.perraudin/darkmattergan/-/blob/master/cosmotools/metric/stats.py
def peak_count(X, neighborhood_size=5, threshold=0.5):
    """
    Peak cound for a 2D or a 3D square image
    :param X: numpy array shape [n,n] or [n,n,n]
    :param neighborhood_size: size of the local neighborhood that should be filtered
    :param threshold: minimum distance betweent the minimum and the maximum to be considered a local maximum
                      Helps remove noise peaks (0.5 since the number of particle is supposed to be an integer)
    :return: vector of peaks found in the array (int)

    How to use
    ----------
    func_pc = partial(peak_count, neighborhood_size=5, threshold=0)
    pcr = np.concatenate( [func_pc(im) for im in imr] )
    pcf = np.concatenate( [func_pc(im) for im in imf] )
    wass_peak = wasserstein_distance_norm(p=pcr, q=pcf)

    """
    size = len(X.shape)
    if len(X.shape) == 1:
        pass
    elif size==2:
        assert(X.shape[0]==X.shape[1])
    elif size==3:
        assert(X.shape[0]==X.shape[1]==X.shape[2])
    else:
        raise Exception(" [!] Too many dimensions")

    # PEAK COUNTS
    data_max = filters.maximum_filter(X, neighborhood_size)
    maxima = (X == data_max)
    if threshold != 0:
        data_min = filters.minimum_filter(X, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

    return np.extract(maxima, X)

# MS-SSIM
def mssim(gen_imgs, gt_imgs):
    """Calculates the MS-SSIM between two sets of images.

    Args:
        gen_imgs (numpy.ndarray): Generated images.
        gt_imgs (numpy.ndarray): Ground-truth images (from simulation).

    Returns:
        float: The MS-SSIM value.
    """
    _gen_imgs = torch.from_numpy(np.expand_dims(gen_imgs, axis=1))
    _gt_imgs = torch.from_numpy(np.expand_dims(gt_imgs, axis=1))
    # TODO: Ensure shape is as expected.
    msssim_val = multiscale_structural_similarity_index_measure(
        _gen_imgs, _gt_imgs,  # Add a dimension for channel to match with torchmetrics expected input.
        gaussian_kernel=True, sigma=1.5, kernel_size=11
    ).item()
    return msssim_val

# Mean density.
def mean_density(img):
    """Calculates mean density of a 2D slice.

    Args:
        img (numpy.ndarray): 2D density slice.

    Returns:
        float: Mean density.
    """
    return img.mean()

# Median density.
def median_density(img):
    """Calculates median density of a 2D slice.

    Args:
        img (numpy.ndarray): 2D density slice.

    Returns:
        float: Median density.
    """
    return np.median(img)

## Cross-correlation coefficient
def correlation_coefficient(delta1, delta2, BoxSize=128):
    """Calculates the cross-correlation coefficient which is a form of normalized cross-power spectrum.
    See equation 6 in https://www.pnas.org/doi/pdf/10.1073/pnas.1821458116 for more details.
    See the corresponding line in Pylians3 source code: `self.r  = self.XPk/np.sqrt(self.Pk[:,0]*self.Pk[:,1])` (https://github.com/franciscovillaescusa/Pylians3/blob/21a33736785ca84dd89a5ac2f73f7b43e981f53d/library/Pk_library/Pk_library.pyx#L1218)

    Args:
        delta1 (numpy.ndarray): generated (or predicted) 2D density slice.
        delta2 (numpy.ndarray): ground-truth 2D density slice.
        BoxSize (float): Simulation box size.

    Returns:
        float: Cross-correlation coefficient.
    """
    # compute cross-power spectrum between two images
    XPk2D = PKL.XPk_plane(delta1, delta2, BoxSize, MAS1='None', MAS2='None', threads=2)

    # get the attributes of the routine
    # k      = XPk2D.k        #k in h/Mpc
    # Pk     = XPk2D.Pk       #auto-Pk of the two maps in (Mpc/h)^2
    # Pk1    = Pk[:,0]        #auto-Pk of the first map in (Mpc/h^2)
    # Pk2    = Pk[:,1]        #auto-Pk of the second map in (Mpc/h^2)
    # XPk    = XPk2D.XPk      #cross-Pk in (Mpc/h)^2
    r      = XPk2D.r        #cross-correlation coefficient
    # Nmodes = XPk2D.Nmodes   #number of modes in each k-bin
    return r

def plot_density(den_gen, den_ip, den_gt):
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, ax = plt.subplots()
    sns.kdeplot(den_gen, ax=ax, color='red', shade=False)
    sns.kdeplot(den_ip, ax=ax, color='blue', shade=False)
    sns.kdeplot(den_gt, ax=ax, color='green', shade=False)
    handles = [
            mpatches.Patch(facecolor=plt.cm.Reds(100), label="cGAN generated"),
            mpatches.Patch(facecolor=plt.cm.Blues(100), label="GR simulation"),
            mpatches.Patch(facecolor=plt.cm.Greens(100), label="f(R) simulation")
        ]
    ax.legend(handles=handles)
    plt.show()

##### Driver function for all evaluation metrics #####
def driver(gens, ips, gts):
    ########################  Run evaluation metrics  ########################
    # 1. PEAK COUNTS
    func_pc = partial(peak_count, neighborhood_size=5, threshold=0.5)
    pc_gen = np.concatenate( [func_pc(im) for im in gens] )
    pc_ip = np.concatenate( [func_pc(im) for im in ips] )
    pc_gt = np.concatenate( [func_pc(im) for im in gts] )
    wass_peak_ip_gen = wasserstein_distance_norm(p=pc_ip, q=pc_gen)
    wass_peak_gt_gen = wasserstein_distance_norm(p=pc_gt, q=pc_gen)
    print(f'Peak count distances:\n\tbetween input GR and generated f(R): {wass_peak_ip_gen}\n\tbetween ground_truth f(R) and generated f(R): {wass_peak_gt_gen}')
    # TODO: Plot?

    del pc_gen, pc_ip, pc_gt, wass_peak_ip_gen, wass_peak_gt_gen
    gc.collect()

    # 2. PIXEL DISTANCE
    wass_pixel_ip_gen = wasserstein_distance_norm(p=ips, q=gens)
    wass_pixel_gt_gen = wasserstein_distance_norm(p=gts, q=gens)
    print(f'Pixel distances:\n\tbetween input GR and generated f(R): {wass_pixel_ip_gen}\n\tbetween ground_truth f(R) and generated f(R): {wass_pixel_gt_gen}')
    # TODO: Plot?

    # 3. MS-SSIM
    # TODO
    # mssim_ip_gen = mssim(val_gen, val_ip)
    # mssim_gt_gen = mssim(val_gen, val_gt)
    # print(f'MS-SSIM:\n\tbetween generated f(R) and input GR: {mssim_ip_gen}\n\tbetween generated f(R) and ground_truth f(R): {mssim_gt_gen}')
    # TODO: Plot?

    # Mean density
    mean_den_gen = np.array([mean_density(im) for im in gens])
    mean_den_ip = np.array([mean_density(im) for im in ips])
    mean_den_gt = np.array([mean_density(im) for im in gts])
    wass_meanden_ip_gen = wasserstein_distance_norm(p=mean_den_ip, q=mean_den_gen)
    wass_meanden_gt_gen = wasserstein_distance_norm(p=mean_den_gt, q=mean_den_gen)
    print(f'Mean density distances:\n\tbetween input GR and generated f(R): {wass_meanden_ip_gen}\n\tbetween ground_truth f(R) and generated f(R): {wass_meanden_gt_gen}')

    plot_density(mean_den_gen, mean_den_ip, mean_den_gt)
    # TODO: Also plot fractional difference below this plot.

    del mean_den_gen, mean_den_ip, mean_den_gt
    gc.collect()

    # Median density
    median_den_gen = np.array([median_density(im) for im in gens])
    median_den_ip = np.array([median_density(im) for im in ips])
    median_den_gt = np.array([median_density(im) for im in gts])
    wass_medianden_ip_gen = wasserstein_distance_norm(p=median_den_ip, q=median_den_gen)
    wass_medianden_gt_gen = wasserstein_distance_norm(p=median_den_gt, q=median_den_gen)
    print(f'Median density distances:\n\tbetween input GR and generated f(R): {wass_medianden_ip_gen}\n\tbetween ground_truth f(R) and generated f(R): {wass_medianden_gt_gen}')

    plot_density(median_den_gen, median_den_ip, median_den_gt)
    # TODO: Also plot fractional difference below this plot.

    del median_den_gen, median_den_ip, median_den_gt
    gc.collect()

    # Correlation coefficient: It is a function of `k`, the wavenumber.
    corr_gen_ip = np.vstack([correlation_coefficient(im_gen, im_ip) for im_gen, im_ip in zip(gens, ips)])
    corr_gen_gt = np.vstack([correlation_coefficient(im_gen, im_gt) for im_gen, im_gt in zip(gens, gts)])

    ax = sns.heatmap(corr_gen_ip, linewidth=0.5, vmin=0., vmax=1.)
    ax.set_title('Correlation coefficient: cGAN-generated f(R) vs Simulation GR')
    plt.show()

    ax = sns.heatmap(corr_gen_gt, linewidth=0.5, vmin=0., vmax=1.)
    ax.set_title('Correlation coefficient: cGAN-generated f(R) vs Simulation f(R)')
    plt.show()

    del corr_gen_ip, corr_gen_gt
    gc.collect()
