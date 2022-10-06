# Evaluation functions

import os
import contextlib
import random
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

gen_gt_color = '#FC9272'
ip_gt_color = '#1C9099'
ip_gen_color = '#2ca25f'

# Power spectrum
def ps_2d(delta, BoxSize=128):
    """Calculates the 2D power spectrum of a density field. It internally calculates the density contrast field and then calculates the power spectrum.

    Args:
        delta (numpy.ndarray): Density slice (note: this is the density field rather than density contrast).
        BoxSize (float): Simulation box size.
    Returns:
        (numpy.ndarray, numpy.ndarray): The wavenumbers and power spectrum amplitudes.
    """
    delta = delta.astype(np.float32)
    # Calculate density contrast.
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    MAS = 'None'
    threads = 2
    Pk2D2 = PKL.Pk_plane(delta, BoxSize, MAS, threads)
    # get the attributes of the routine
    k2      = Pk2D2.k      #k in h/Mpc
    Pk2     = Pk2D2.Pk     #Pk in (Mpc/h)^2
#     Nmodes = Pk2D2.Nmodes #Number of modes in the different k bins
    return k2, Pk2

def chiq_squared_dist_ps(ps, ps_expected):
    """Calculate chi-squared distance between power spectra.

    Args:
        ps (numpy.ndarray): Calculated (or observed) power spectrum.
        ps_expected (numpy.ndarray): Ground-truth (or expected) power spectrum.

    Returns:
        float: Chi-square distance
    
    Notes
    -----
    See, for example, http://www.cs.columbia.edu/~mmerler/project/code/pdist2.m

    """
    return 0.5 * np.sum((ps-ps_expected)**2 / (ps+ps_expected))

assert np.allclose(chiq_squared_dist_ps(np.ones(10), np.ones(10)), 0.0)

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
def mssim_single(gen_img, gt_img):
    """Calculates the MS-SSIM between a sets of images.
    Args:
        gen_img (numpy.ndarray): Generated image.
        gt_img (numpy.ndarray): Ground-truth image (from simulation).
    Returns:
        float: MS-SSIM value.

    """
    assert gen_img.shape == gt_img.shape

    # gen_img, for example, is of shape (512, 512).

    msssim_val = multiscale_structural_similarity_index_measure(
        torch.from_numpy(gen_img).unsqueeze(0).unsqueeze(0), torch.from_numpy(gt_img).unsqueeze(0).unsqueeze(0),  # Add a dimension for channel to match with torchmetrics expected input.
        gaussian_kernel=True, sigma=1.5, kernel_size=11
    ).item()

    return msssim_val

# MS-SSIM (multiple)
def mssim_multiple(gen_imgs, gt_imgs):
    """Calculates the MS-SSIM between multiple sets of images.
    Args:
        gen_imgs (numpy.ndarray): Generated images.
        gt_imgs (numpy.ndarray): Ground-truth images (from simulation).
    Returns:
        numpy.ndarray: One-dimensional array of MS-SSIM values.
    
    Note
    ----
    TODO: This function is not tested as of now.

    """
    assert gen_imgs.shape == gt_imgs.shape

    _gen_imgs = torch.from_numpy(np.expand_dims(gen_imgs, axis=1))
    _gt_imgs = torch.from_numpy(np.expand_dims(gt_imgs, axis=1))
    # TODO: Ensure shape is as expected.
    msssim_arr = np.empty(gen_imgs.shape)
    for i in range(len(gen_imgs)):
        msssim_val = multiscale_structural_similarity_index_measure(
            _gen_imgs[i].unsqueeze(0), _gt_imgs[i].unsqueeze(0),  # Add a dimension for channel to match with torchmetrics expected input.
            gaussian_kernel=True, sigma=1.5, kernel_size=11
        ).item()
        msssim_arr[i] = msssim_val
    return msssim_arr

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
        r (float): Cross-correlation coefficient.
        k (numpy.ndarray): Wavenumbers.
    """
    delta1 = delta1.astype(np.float32)
    delta2 = delta2.astype(np.float32)

    # compute cross-power spectrum between two images
    XPk2D = PKL.XPk_plane(delta1, delta2, BoxSize, MAS1='None', MAS2='None', threads=2)

    # get the attributes of the routine
    k      = XPk2D.k        #k in h/Mpc
    # Pk     = XPk2D.Pk       #auto-Pk of the two maps in (Mpc/h)^2
    # Pk1    = Pk[:,0]        #auto-Pk of the first map in (Mpc/h^2)
    # Pk2    = Pk[:,1]        #auto-Pk of the second map in (Mpc/h^2)
    # XPk    = XPk2D.XPk      #cross-Pk in (Mpc/h)^2
    r      = XPk2D.r        #cross-correlation coefficient
    # Nmodes = XPk2D.Nmodes   #number of modes in each k-bin
    return r, k

def transfer_function(ps_pred, ps_true):
    return np.sqrt(ps_pred/ps_true)

# def plot_mat(corr_mat, k, title=None):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     plt.rcParams['figure.figsize'] = [20, 8]
#     fig, ax = plt.subplots()
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     mat = ax.matshow(corr_mat)
#     fig.colorbar(mat, cax=cax, orientation='vertical')
#     ax.set_title(title)
#     ax.set_xticks(np.arange(len(k)))
#     ax.set_xticklabels(k)
#     plt.show()

def plot_density(den_gen, den_ip, den_gt, plotting_mean=True):  # plotting_mean is only used for setting the plot title. If False, it is assumed we are plotting the median density.
    # Note that the KDE plots, similar to histograms, are trying to approximate the PDF that generated the values shown in the plot.
    # See more here: https://seaborn.pydata.org/tutorial/distributions.html#tutorial-kde
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0)
    sns.kdeplot(den_gen, ax=ax[0], shade=False, x='cosmological density', y=None, color=gen_gt_color)
    sns.kdeplot(den_ip, ax=ax[0], shade=False, c=ip_gt_color)
    sns.kdeplot(den_gt, ax=ax[0], shade=False, c='black')
    if plotting_mean:
        ax[0].set_title('Cosmological mean density distribution')
    else:
        ax[0].set_title('Cosmological median density distribution')
    ax[0].set_ylabel('$N_{pixels}$')
    handles = [
            mpatches.Patch(facecolor=gen_gt_color, label="cGAN generated"),
            mpatches.Patch(facecolor=ip_gt_color, label="GR simulation"),
            mpatches.Patch(facecolor='black', label="f(R) simulation")
        ]
    ax[0].legend(handles=handles)

    ax[1].set_xscale('log')
    ax[1].plot(100 * (den_gt - den_gen) / den_gt, c=gen_gt_color)
    ax[1].plot(100 * (den_gt - den_ip) / den_gt, c=ip_gt_color)
    ax[1].axhline(y=0, linestyle='--', c='black')
    ax[1].set_ylabel('Relative difference (%)', fontsize=14)
    ax[1].set_xlabel('pixel value (cosmological density)', fontsize=14);
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    plt.show()

    plt.show()

def frac_diff(real, fake):
    return np.abs(real - fake) / real

##### Driver function for all evaluation metrics #####
def driver(gens, ips, gts):
    # TODO: Also show shaded errorbars around plots since all are averaged over multiple images.

    ########################  Run evaluation metrics  ########################
    # 1. AVERAGED POWER SPECTRUM, TRANSFER FUNCTION, AND CORRELATION COEFFICIENT
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # Prevent unnecessary verbose output from printing on screen.
        k = ps_2d(gens[0])[0]
        ps_gen = np.vstack([ps_2d(im)[1] for im in gens]).mean(axis=0)
        ps_ip = np.vstack([ps_2d(im)[1] for im in ips]).mean(axis=0)
        ps_gt = np.vstack([ps_2d(im)[1] for im in gts]).mean(axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0)

    ax[0].loglog(k, ps_gen, c=gen_gt_color, label='cGAN generated')
    ax[0].loglog(k, ps_ip, c=ip_gt_color, label='GR simulation')
    ax[0].loglog(k, ps_gt, c='black', label='f(R) simulation')
    ax[0].legend()
    ax[0].set_title('Averaged Power Spectrum', fontsize=18)
    ax[0].tick_params(axis='x', labelsize=12)
    ax[0].tick_params(axis='y', labelsize=12)

    ax[1].set_xscale('log')
    ax[1].plot(k, 100 * (ps_gt - ps_gen) / ps_gt, c=gen_gt_color)
    ax[1].plot(k, 100 * (ps_gt - ps_ip) / ps_gt, c=ip_gt_color)
    ax[1].axhline(y=0, c='black', linestyle='--')
    ax[1].set_ylabel('Relative difference (%)', fontsize=14)
    ax[1].set_xlabel('k (h/Mpc)', fontsize=14);
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].fill_between(k, -25, 25, alpha=0.2)
    ax[0].set_xlim([k.min(), 15.])
    ax[1].set_xlim([k.min(), 15.])
    ax[1].set_ylim([-50, 50])

    #### Repeat the relative difference plot as above but taking input GR as reference ####
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(hspace=0)

    ax.set_xscale('log')
    ax.plot(k, 100 * (ps_ip - ps_gen) / ps_ip, c=ip_gen_color, label='cGAN generated')
    ax.plot(k, 100 * (ps_ip - ps_gt) / ps_ip, c=ip_gt_color, label='f(R) simulation')
    ax.axhline(y=0, c='black', linestyle='--')
    ax.set_ylabel('Relative difference (%)', fontsize=14)
    ax.set_xlabel('k (h/Mpc)', fontsize=14);
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.fill_between(k, -25, 25, alpha=0.2)
    ax.set_xlim([k.min(), 15.])
    ax.set_xlim([k.min(), 15.])
    ax.set_ylim([-50, 50])
    ax.set_title('Reference is GR simulation')
    ax.legend()
    #######################################################################################

    # Now plot transfer function and stochasticity.
    fig, ax = plt.subplots(2, 1, figsize=(18, 18))

    ax[0].plot(k, transfer_function(ps_gen, ps_gt), label='cGAN generated', c=gen_gt_color)
    ax[0].plot(k, transfer_function(ps_ip, ps_gt), label='GR simulation', c=ip_gt_color)
    ax[0].set_xscale('log')
    ax[0].axhline(y=1., c='black', linestyle='--')
    ax[0].set_ylabel('$T(k)$')
    ax[0].set_xlabel('$k [h/Mpc]$')
    ax[0].set_title('Transfer function')
    ax[0].legend()

    # Correlation coefficient: It is a function of `k`, the wavenumber.
    # Get wavenumbers
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        k = correlation_coefficient(gens[0], ips[0])[1]
        corr_gen_gt = np.vstack([correlation_coefficient(im_gen, im_gt)[0] for im_gen, im_gt in zip(gens, ips)]).mean(axis=0)
        corr_ip_gt = np.vstack([correlation_coefficient(im_ip, im_gt)[0] for im_ip, im_gt in zip(ips, gts)]).mean(axis=0)

    ax[1].plot(k, 1 - corr_gen_gt ** 2, label='cGAN generated', c=gen_gt_color)
    ax[1].plot(k, 1 - corr_ip_gt ** 2, label='GR simulation', c=ip_gt_color)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_ylabel('$1 - r(k)^2$')
    ax[1].set_xlabel('$k [h/Mpc]$')
    ax[1].set_title('Stochasticity')
    ax[1].legend()

    plt.show()

#     plot_mat(corr_gen_ip, k, title='Cross-correlation coefficient: cGAN-generated f(R) vs Simulation GR')
#     plot_mat(corr_gen_gt, k, title='Cross-correlation coefficient: cGAN-generated f(R) vs Simulation f(R)')

#     ax = sns.heatmap(corr_gen_ip, linewidth=0.5, xticklabels=False, yticklabels=False, vmin=-1., vmax=1.)
#     ax.set_title('Correlation coefficient: cGAN-generated f(R) vs Simulation GR')
#     plt.show()

#     ax = sns.heatmap(corr_gen_gt, linewidth=0.5, xticklabels=False, yticklabels=False, vmin=-1., vmax=1.)
#     ax.set_title('Correlation coefficient: cGAN-generated f(R) vs Simulation f(R)')
#     plt.show()

    # Print chi-squared distance between averaged power spectra.
    chisquare_ip_gt = chiq_squared_dist_ps(ps_ip, ps_gt)
    chisquare_gen_gt = chiq_squared_dist_ps(ps_gen, ps_gt)
    print(f'Chi-squared distance between averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt}')

    del corr_gen_gt, corr_ip_gt, ps_gen, ps_ip, ps_gt
    gc.collect()

    # 2. PEAK COUNTS
    func_pc = partial(peak_count, neighborhood_size=5, threshold=0.5)

#     pc_gen = np.vstack( [func_pc(im) for im in gens] ).mean(axis=0)
#     pc_ip = np.vstack( [func_pc(im) for im in ips] ).mean(axis=0)
#     pc_gt = np.vstack( [func_pc(im) for im in gts] ).mean(axis=0)
#     fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
#     fig.subplots_adjust(hspace=0)

#     sns.kdeplot(pc_gen, ax=ax[0], shade=False, x='pixel value', y=None, color=gen_gt_color)
#     sns.kdeplot(pc_ip, ax=ax[0], shade=False, x='pixel value', y=None, color=ip_gt_color)
#     sns.kdeplot(pc_gt, ax=ax[0], shade=False, x='pixel value', y=None, color='black')

#     ax[1].set_xscale('log')
#     ax[1].plot(100 * (pc_gt - pc_gen) / pc_gt, c=gen_gt_color)
#     ax[1].plot(100 * (pc_gt - pc_ip) / pc_gt, c=ip_gt_color)
#     ax[1].plot(100 * (pc_gt - pc_gt) / pc_gt, c='black')
#     ax[1].set_ylabel('Relative difference (%)', fontsize=14)
#     ax[1].set_xlabel('k (h/Mpc)', fontsize=14);
#     ax[1].tick_params(axis='x', labelsize=12)
#     ax[1].tick_params(axis='y', labelsize=12)
#     plt.show()

#     del pc_gen, pc_ip, pc_gt

    pc_gen = np.concatenate( [func_pc(im) for im in gens] )
    pc_ip = np.concatenate( [func_pc(im) for im in ips] )
    pc_gt = np.concatenate( [func_pc(im) for im in gts] )
    wass_peak_gt_ip = wasserstein_distance_norm(p=pc_gt, q=pc_ip)
    wass_peak_gt_gen = wasserstein_distance_norm(p=pc_gt, q=pc_gen)
    print(f'Peak count distances:\n\tbetween ground truth f(R) and input GR: {wass_peak_gt_ip}\n\tbetween ground_truth f(R) and generated f(R): {wass_peak_gt_gen}')
    # TODO: Plot?

    del pc_gen, pc_ip, pc_gt, wass_peak_gt_ip, wass_peak_gt_gen
    gc.collect()

    # 3. PIXEL DISTANCE
#     pixel_gen = np.vstack( [func_pc(im) for im in gens] ).mean(axis=0)
#     pixel_ip = np.vstack( [func_pc(im) for im in ips] ).mean(axis=0)
#     pixel_gt = np.vstack( [func_pc(im) for im in gts] ).mean(axis=0)
#     fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
#     fig.subplots_adjust(hspace=0)

#     sns.kdeplot(pixel_gen, ax=ax[0], shade=False, x='pixel value', y=None, color=gen_gt_color)
#     sns.kdeplot(pixel_ip, ax=ax[0], shade=False, x='pixel value', y=None, color=ip_gt_color)
#     sns.kdeplot(pixel_gt, ax=ax[0], shade=False, x='pixel value', y=None, color='black')

#     ax[1].set_xscale('log')
#     ax[1].plot(100 * (pixel_gt - pixel_gen) / ps_gt, c=gen_gt_color)
#     ax[1].plot(100 * (pixel_gt - pixel_ip) / ps_gt, c=ip_gt_color)
#     ax[1].plot(100 * (pixel_gt - pixel_gt) / ps_gt, c='black')
#     ax[1].set_ylabel('Relative difference (%)', fontsize=14)
#     ax[1].set_xlabel('k (h/Mpc)', fontsize=14);
#     ax[1].tick_params(axis='x', labelsize=12)
#     ax[1].tick_params(axis='y', labelsize=12)
#     plt.show()

#     del pixel_gen, pixel_ip, pixel_gt

    ####################################################################################################################################################################################
    ### The reason why evaluate distance on chunks of images rather than all images is because all images do not fit into memory due to which the RAM (~13 GB in Kaggle kernel) blows up.
    ####################################################################################################################################################################################
    wass_pixel_gt_ips = []
    wass_pixel_gt_gens = []
    for index in [0, 50, 100, 150, 200, 250, 300, 350, 400]:
        if index == 400:
            assert len(gts[index:]) == 61
            wass_pixel_gt_ip = wasserstein_distance_norm(p=gts[index:], q=ips[index:])
            wass_pixel_gt_gen = wasserstein_distance_norm(p=gts[index:], q=gens[index:])
        else:
            assert len(gts[index:index+50]) == 50
            wass_pixel_gt_ip = wasserstein_distance_norm(p=gts[index:index+50], q=ips[index:index+50])
            wass_pixel_gt_gen = wasserstein_distance_norm(p=gts[index:index+50], q=gens[index:index+50])

        wass_pixel_gt_ips.append(wass_pixel_gt_ip)
        wass_pixel_gt_gens.append(wass_pixel_gt_gen)

    mean_wass_pixel_gt_ip = np.mean(wass_pixel_gt_ips)
    mean_wass_pixel_gt_gen = np.mean(wass_pixel_gt_gens)

    print(f'Pixel distances:\n\tbetween ground truth f(R) and input GR: {mean_wass_pixel_gt_ip}\n\tbetween ground_truth f(R) and generated f(R): {mean_wass_pixel_gt_gen}')

    # 4. MS-SSIM
    # Some motivation for the idea: https://dl.acm.org/doi/pdf/10.5555/3305890.3305954
    # Specific approach motivated from https://arxiv.org/pdf/2004.08139.pdf
    def pair_generator(population_size):  # From https://stackoverflow.com/a/50531575
        random.seed(42)
        pop_range = 2 * population_size
        population = [i for i in range(1, pop_range + 1)]
        random.shuffle(population)
        for _ in range(population_size):
            yield [population.pop(), population.pop()]

    gen_msssims = []
    gt_msssims = []
    ip_msssims = []

    for index_pair in pair_generator(int(len(gens)/2) - 1):
        i1, i2 = index_pair[0], index_pair[1]
        val = mssim_single(gens[i1], gens[i2])
        gen_msssims.append(val)

        val = mssim_single(gts[i1], gts[i2])
        gt_msssims.append(val)

        val = mssim_single(ips[i1], ips[i2])
        ip_msssims.append(val)

    gen_gt_significance = (np.mean(gen_msssims) - np.mean(gt_msssims)) / ((np.std(gen_msssims) + np.std(gt_msssims)) / 2)
    ip_gt_significance = (np.mean(ip_msssims) - np.mean(gt_msssims)) / ((np.std(ip_msssims) + np.std(gt_msssims)) / 2)
    print(f'MS-SSIM significance:\n\tbetween cGAN generated and ground truth f(R): {gen_gt_significance}\n\tbetween input GR and ground truth f(R): {ip_gt_significance}')

    # TODO: Plot?

    # 5. MEAN DENSITY
    mean_den_gen = np.array([mean_density(im) for im in gens])
    mean_den_ip = np.array([mean_density(im) for im in ips])
    mean_den_gt = np.array([mean_density(im) for im in gts])
    wass_meanden_gt_ip = wasserstein_distance_norm(p=mean_den_gt, q=mean_den_ip)
    wass_meanden_gt_gen = wasserstein_distance_norm(p=mean_den_gt, q=mean_den_gen)
    print(f'Mean density distances:\n\tbetween ground truth f(R) and input GR: {wass_meanden_gt_ip}\n\tbetween ground_truth f(R) and generated f(R): {wass_meanden_gt_gen}')

    plot_density(mean_den_gen, mean_den_ip, mean_den_gt, plotting_mean=True)

    del mean_den_gen, mean_den_ip, mean_den_gt, wass_meanden_gt_ip, wass_meanden_gt_gen
    gc.collect()

    # 6. MEDIAN DENSITY
    median_den_gen = np.array([median_density(im) for im in gens])
    median_den_ip = np.array([median_density(im) for im in ips])
    median_den_gt = np.array([median_density(im) for im in gts])
    wass_medianden_gt_ip = wasserstein_distance_norm(p=median_den_gt, q=median_den_ip)
    wass_medianden_gt_gen = wasserstein_distance_norm(p=median_den_gt, q=median_den_gen)
    print(f'Median density distances:\n\tbetween ground truth f(R) and input GR: {wass_medianden_gt_ip}\n\tbetween ground_truth f(R) and generated f(R): {wass_medianden_gt_gen}')

    plot_density(median_den_gen, median_den_ip, median_den_gt, plotting_mean=False)

    del median_den_gen, median_den_ip, median_den_gt, wass_medianden_gt_ip, wass_medianden_gt_gen
    gc.collect()
