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

sns.set_context("paper", font_scale = 2)
sns.set_style('whitegrid')
sns.set(style='ticks')

import Pk_library.Pk_library as PKL

from scipy.stats import wasserstein_distance
import scipy.ndimage.filters as filters

import torch
from torchmetrics.functional import multiscale_structural_similarity_index_measure

from scipy.stats import ks_2samp
from scipy.stats import median_abs_deviation, iqr

gen_gt_color = '#FC9272'
ip_gt_color = '#1C9099'
ip_gen_color = '#2ca25f'

ticklabelsize = 17
axeslabelsize = 19
titlesize = 20

#from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
#def smooth_ps(k, Pk):
#    w = np.linspace(0.01, 1, k.shape[0])
#    spl = UnivariateSpline(k, Pk, w=w)
#    return k, spl(k)

# Below function taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html#smoothing-of-a-1d-signal
def smooth(x,window_len=5,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[round(window_len/2-1):-round(window_len/2)]

# Power spectrum
def ps_2d(delta, BoxSize=128, vel_field=False):
    """Calculates the 2D power spectrum of a density/velocity divergence field. It internally calculates the density contrast field and then calculates the power spectrum. For velocity divergence, the field kept as it is.

    Args:
        delta (numpy.ndarray): Density slice (note: this is the density field rather than density contrast).
        BoxSize (float): Simulation box size.
    Returns:
        (numpy.ndarray, numpy.ndarray): The wavenumbers and power spectrum amplitudes.
    """
    delta = delta.astype(np.float32)
    if not vel_field:
        # Calculate density contrast.
        delta = delta / np.mean(delta, dtype=np.float64); delta = delta - 1.0

    MAS = 'None'
    threads = 2
    Pk2D2 = PKL.Pk_plane(delta, BoxSize, MAS, threads)
    # get the attributes of the routine
    k2      = Pk2D2.k      #k in h/Mpc
    Pk2     = Pk2D2.Pk     #Pk in (Mpc/h)^2
#     Nmodes = Pk2D2.Nmodes #Number of modes in the different k bins
    return k2, Pk2

def chiq_squared_dist_ps(ps, ps_expected, ps_expected_std, num_images):  # See https://stats.stackexchange.com/questions/184101/comparing-two-histograms-using-chi-square-distance: many interpretations for the chi-squared distance exist. The form used below has the advantage tht it is symmetric wrt the variables.
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
    return (1/(num_images - 1)) * np.sum(((ps-ps_expected) ** 2) / (ps_expected_std ** 2))

def ps_metric(ps, ps_expected):
    return 100 * (ps / ps_expected - 1)

#assert np.allclose(chiq_squared_dist_ps(np.ones(10), np.ones(10), np.zeros(10), 100), 0.0)

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

"""
# Peak count.
# Code taken from https://renkulab.io/gitlab/nathanael.perraudin/darkmattergan/-/blob/master/cosmotools/metric/stats.py
def peak_count(X, neighborhood_size=5, threshold=0.5, remove_outliers=False):
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
    
    p = np.extract(maxima, X)
    if remove_outliers:
        p = p[p < np.quantile(p, 0.99)]

    return p
"""

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

def plot_density(den_gen, den_ip, den_gt, plotting_mean=True, vel_field=True):  # plotting_mean is only used for setting the plot title. If False, it is assumed we are plotting the median density.
    # Note that the KDE plots, similar to histograms, are trying to approximate the PDF that generated the values shown in the plot.
    # See more here: https://seaborn.pydata.org/tutorial/distributions.html#tutorial-kde
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0)
    sns.kdeplot(den_gen, ax=ax[0], fill=False, y=None, color=gen_gt_color)
    sns.kdeplot(den_ip, ax=ax[0], fill=False, c=ip_gt_color)
    sns.kdeplot(den_gt, ax=ax[0], fill=False, c='black')
    if plotting_mean:
        if vel_field:
            ax[0].set_title('Velocity divergence distribution')
        else:
            ax[0].set_title('Cosmological mean density distribution')
    else:
        if vel_field:
            ax[0].set_title('Velocity divergence distribution')
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
    if vel_field:
        ax[1].set_xlabel('pixel value (velocity divergence)', fontsize=14);    
    else:
        ax[1].set_xlabel('pixel value (cosmological density)', fontsize=14);
    ax[1].tick_params(axis='x', labelsize=12)
    ax[1].tick_params(axis='y', labelsize=12)
    plt.savefig(f'density_compare_{plotting_mean}.png')

    plt.show()

def frac_diff(real, fake):
    return np.abs(real - fake) / real

import numpy as np
import functools
import multiprocessing as mp

import scipy.ndimage.filters as filters

"""
def peak_count_hist(dat, bins=20, lim=None, neighborhood_size=5, threshold=0, log=True, mean=True, remove_outliers=False):
    
    # Remove single dimension...
    dat = np.squeeze(dat)
    
    #num_workers = mp.cpu_count() - 1
    num_workers = 1
    with mp.Pool(processes=num_workers) as pool:
        peak_count_arg = functools.partial(peak_count, neighborhood_size=neighborhood_size, threshold=threshold, remove_outliers=remove_outliers)
        peak_arr = np.array(pool.map(peak_count_arg, dat))
    peak = np.hstack(peak_arr)
    if log:
        peak = np.log(peak+np.e)
        peak_arr = np.array([np.log(pa+np.e) for pa in peak_arr])
    if lim is None:
        lim = (np.min(peak), np.max(peak))
    else:
        lim = tuple(map(type(peak[0]), lim))
    # Compute histograms individually
    with mp.Pool(processes=num_workers) as pool:
        hist_func = functools.partial(unbounded_histogram, bins=bins, range=lim)
        res = np.array(pool.map(hist_func, peak_arr))
    
    # Unpack results
    y = np.vstack(res[:, 0])
    x = res[0, 1]

    x = (x[1:] + x[:-1]) / 2
    if log:
        x = np.exp(x)-np.e
    if mean:
        y = np.mean(y, axis=0)
    return y, x, lim


def unbounded_histogram(dat, range=None, remove_outliers=False, **kwargs):
    if remove_outliers:
        dat = dat.ravel()
        dat = dat[dat < np.quantile(dat, 0.99)]
    if range is None:
        return np.histogram(dat, **kwargs)
    y, x = np.histogram(dat, range=range, **kwargs)
    y[0] = y[0] + np.sum(dat<range[0])
    y[-1] = y[-1] + np.sum(dat>range[1])
    return y, x

def peak_count_hist_real_fake(real, fake, bins=20, lim=None, log=True, neighborhood_size=5, threshold=0, mean=True, remove_outliers=False):
    y_real, x, lim = peak_count_hist(real, bins=bins, lim=None, log=log, neighborhood_size=neighborhood_size, threshold=threshold, mean=mean, remove_outliers=remove_outliers)
    y_fake, _, _ = peak_count_hist(fake, bins=bins, lim=None, log=log, neighborhood_size=neighborhood_size, threshold=threshold, mean=mean, remove_outliers=remove_outliers)
    return y_real, y_fake, x

def mass_hist(dat, bins=20, lim=None, log=True, mean=True, remove_outliers=False, **kwargs):
    if log:
        log_data = np.log10(dat + 1)
    else:
        log_data = dat
    if lim is None:
        lim = (np.min(log_data), np.max(log_data))

    #num_workers = mp.cpu_count() - 1
    num_workers = 1
    with mp.Pool(processes=num_workers) as pool:
        results = [pool.apply(unbounded_histogram, (x,), dict(bins=bins, range=lim, remove_outliers=remove_outliers)) for x in log_data]
    y = np.vstack([y[0] for y in results])
    x = results[0][1]
    if log:
        x = 10**((x[1:] + x[:-1]) / 2) - 1
    else:
        x = (x[1:] + x[:-1]) / 2
    if mean:
        return np.mean(y, axis=0), x, lim
    else:
        return y, x, lim

def mass_hist_real_fake(real, fake, bins=20, lim=None, log=True, mean=True, remove_outliers=False):
    if lim is None:
        new_lim = True
    else:
        new_lim = False
    y_real, x, lim = mass_hist(real, bins=bins, lim=None, log=log, mean=mean, remove_outliers=remove_outliers)
    if new_lim:
        lim = list(lim)
        lim[1] = lim[1]+1
        y_real, x, lim = mass_hist(real, bins=bins, lim=None, log=log, mean=mean, remove_outliers=remove_outliers)

    y_fake, _, _ = mass_hist(fake, bins=bins, lim=None, log=log, mean=mean, remove_outliers=remove_outliers)
    return y_real, y_fake, x
"""


####################### New code ###########################
# Custom pixel PDF code.


import multiprocessing as mp
import smoothing_library as SL

def pixel_hist(image, bins=np.logspace(start=-2, stop=2, num=99)):  # bins are defined inside the function.
    # First smooth the field.
    field = image.astype(np.float32)
    BoxSize = 128.0 #Mpc/h
    grid    = field.shape[0]
    Filter  = 'Top-Hat'
    threads = 1
    #kmin    = 0  #h/Mpc
    #kmax    = 10 #h/Mpc

    R = 10  # arbitrarily chosen by me.

    # compute the filter in Fourier space
    W_k = SL.FT_filter_2D(BoxSize, R, grid, Filter, threads)
    # smooth the field
    field_smoothed = SL.field_smoothing_2D(field, W_k, threads)

    field_smoothed = field_smoothed / field_smoothed.mean()

    #bins = np.logspace(start=-2, stop=2, num=99)  # Bins are set according to https://browse.arxiv.org/pdf/2109.02636.pdf
    
    counts, bin_edges = np.histogram(field_smoothed, bins=bins)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bincenters, counts

def pixel_pdf(images, bins=np.logspace(start=-2, stop=2, num=99)):
    #images is of shape: num_examples x height x width
    #num_workers = mp.cpu_count() - 1
    pixel_hist_y, pixel_hist_x = [], []
    for x in images:
        bincenters, counts = pixel_hist(x, bins=bins)
        pixel_hist_x.append(bincenters)
        pixel_hist_y.append(counts)

    pixel_hist_x = np.vstack(pixel_hist_x)
    pixel_hist_y = np.vstack(pixel_hist_y)

    #num_workers = 2
    #with mp.Pool(processes=num_workers) as pool:
    #    results = np.array([pool.apply(pixel_hist, (x,)) for x in images])

    #pixel_hist_y = np.vstack([y[1] for y in results])
    #pixel_hist_x = np.vstack([y[0] for y in results])
    #pixel_hist_x = np.exp(pixel_hist_x) if log else pixel_hist_x

    #x = results[0][0]

    x = np.mean(pixel_hist_x, axis=0)
    y = np.mean(pixel_hist_y, axis=0)
    xstd = np.std(pixel_hist_x, axis=0)
    ystd = np.std(pixel_hist_y, axis=0)
    ymedian = np.median(pixel_hist_y, axis=0)
    ymad = iqr(pixel_hist_y, axis=0)
    xmedian = np.median(pixel_hist_x, axis=0)
    xmad = iqr(pixel_hist_x, axis=0)
    return x, y, xstd, ystd, ymedian, ymad, xmedian, xmad

import smoothing_library as SL
import numpy as np
import matplotlib.pyplot as plt

def cumulants(field, R=10, n=1):  # n=2 means variance, n=3 means skewness, n=4 means kurtosis
  # Compute density contrast
  field = field / np.mean(field, dtype=np.float64); field = field - 1.0

  field = field.astype(np.float32)
 
  BoxSize = 128.0 #Mpc/h
  grid    = field.shape[0]
  Filter  = 'Top-Hat'
  threads = 1
  #kmin    = 0  #h/Mpc
  #kmax    = 10 #h/Mpc

  # compute the filter in Fourier space
  W_k = SL.FT_filter_2D(BoxSize, R, grid, Filter, threads)
  # smooth the field
  field_smoothed = SL.field_smoothing_2D(field, W_k, threads)

  central_moment = (1 / grid**2) * np.sum((field_smoothed - field_smoothed.mean()) ** n)

  # Now calculate cumulant
  if n == 2 or n == 3:
    return central_moment
  elif n == 4:
    cumulant_n2 = (1 / grid**2) * np.sum((field_smoothed - field_smoothed.mean()) ** 2)
    return central_moment - 3 * cumulant_n2**2

def calculate_cumulant(field, n=2):
  rth = np.arange(1.01, 12.8, 0.5)  # limits taken from sec 4.1.4 of https://arxiv.org/pdf/1305.7486.pdf. But the lower limit is purposefully set to be ~ 1.
  cs = []
  for R in rth:
    cs.append(
        cumulants(field, R=R, n=n)
    )
  return rth, cs

def cumulant_overall(imgs, n=2):
    cs_combined = []
    rth_combined = []
    for img in imgs:
        rth, cs = calculate_cumulant(img, n=n)
        rth_combined.append(rth)
        cs_combined.append(cs)

    x = np.log10(np.mean(rth_combined, axis=0))
    x_std = np.log10(np.std(rth_combined, axis=0))
    y = np.log10(np.mean(cs_combined, axis=0))
    ystd = np.log10(np.std(cs_combined, axis=0))
    ymad = np.log10(iqr(cs_combined, axis=0))
    ymedian = np.log10(np.median(cs_combined, axis=0))
    xmad = np.log10(iqr(rth_combined, axis=0))
    xmedian = np.log10(np.median(rth_combined, axis=0))
    return x, y, x_std, ystd, ymedian, ymad, xmedian, xmad

def compute_density_contrast(field):
    return field / field.mean() - 1

###########################################################
##### Driver function for all evaluation metrics #####
def driver(gens, ips, gts, vel_field=False, name=None, val_or_test=0,epoch_num=None,field_name='F4_den', save=True):  # In val_or_test, 0 means val set and 1 means test set.
    ########################  Run evaluation metrics  ########################
    # 1. AVERAGED POWER SPECTRUM, TRANSFER FUNCTION, AND CORRELATION COEFFICIENT
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):  # Prevent unnecessary verbose output from printing on screen.
        k = ps_2d(gens[0], vel_field=vel_field)[0]
        gen__ = np.vstack([ps_2d(im, vel_field=vel_field)[1][k <= 5] for im in gens])
        ip__ = np.vstack([ps_2d(im, vel_field=vel_field)[1][k <= 5] for im in ips])
        gt__ = np.vstack([ps_2d(im, vel_field=vel_field)[1][k <= 5] for im in gts])

        # Select only upto k = 5 since scales smaller than that are not well reproduced by MG-GLAM itself. So using them does not make sense.
        k = k[k <= 5]

        ps_gen = gen__.mean(axis=0)
        ps_ip = ip__.mean(axis=0)
        ps_gt = gt__.mean(axis=0)
        ps_gen_std = np.std(gen__, axis=0)
        ps_ip_std = np.std(ip__, axis=0)
        ps_gt_std = np.std(gt__, axis=0)
        ps_gen_iqr = iqr(gen__, axis=0)
        ps_ip_iqr = iqr(ip__, axis=0)
        ps_gt_iqr = iqr(gt__, axis=0)
        print(f'Averaged generated power spectra standard deviation: {ps_gen_iqr}')
   #
    # Median power spectrum.
    ps_gen_median = np.median(gen__, axis=0)
    ps_ip_median = np.median(ip__, axis=0)
    ps_gt_median = np.median(gt__, axis=0)

    assert ips.shape[0] == gts.shape[0] == gens.shape[0]
    num_images = gts.shape[0]

    print("Median power spectrum...")
    # We check distances at two different regimes: linear/quasi-linear and non-linear. Any scale, k, greater than 10 * (2*pi/L_box) is considered non-linear. It's just a rough estimate taken from https://iopscience.iop.org/article/10.1088/0004-637X/724/2/878: "For simulations with Lbox < 200 h−1 Mpc, nonlinearity has set in at k ≳ 10 × 2π/Lbox".
    print("1.....(k < 0.5)")
    inds1 = (k <= 0.5)
    chisquare_ip_gt_median = ps_metric(ps_ip_median[inds1], ps_gt_median[inds1]).mean()
    chisquare_gen_gt_median = ps_metric(ps_gen_median[inds1], ps_gt_median[inds1]).mean()
    print(f'Distance between averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_median}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_median}')
    print("2..... (k > 0.5)")
    inds2 = (k > 0.5)
    chisquare_ip_gt_median = ps_metric(ps_ip_median[inds2], ps_gt_median[inds2]).mean()
    chisquare_gen_gt_median = ps_metric(ps_gen_median[inds2], ps_gt_median[inds2]).mean()
    print(f'Distance between averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_median}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_median}')

    # Print chi-squared distance between averaged power spectra.
    print("Mean power spectrum...")
    print("1.....(k <= 0.5)")
    chisquare_ip_gt = ps_metric(ps_ip[inds1], ps_gt[inds1]).mean()
    chisquare_gen_gt = ps_metric(ps_gen[inds1], ps_gt[inds1]).mean()
    print(f'Distance between averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt}')

    print("2.....(k > 0.5)")
    chisquare_ip_gt = ps_metric(ps_ip[inds2], ps_gt[inds2]).mean()
    chisquare_gen_gt = ps_metric(ps_gen[inds2], ps_gt[inds2]).mean()
    print(f'Distance between averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt}')

    # Print standard deviation of averaged power spectrum (i.e., scatter).
    #chisquare_ip_gt_mad = chiq_squared_dist_ps(ps_ip_mad, ps_gt_mad)
    #chisquare_gen_gt_mad = chiq_squared_dist_ps(ps_gen_mad, ps_gt_mad)
    #print(f'Chi-squared distance between the standard deviation of averaged power spectra:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_mad}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_mad}')

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].loglog(k, ps_gen_median, c=gen_gt_color, label=fr'cGAN: Dist = {chisquare_gen_gt_median:.2f}')
        ax[0].loglog(k, ps_ip_median, c=ip_gt_color, label=fr'GR sim: Dist = {chisquare_ip_gt_median:.2f}')
        ax[0].loglog(k, ps_gt_median, c='black', label='f(R) sim')
        ax[0].legend()
        ax[0].set_title('Median power spectrum', fontsize=titlesize)
        ax[0].tick_params(axis='x', labelsize=ticklabelsize)
        ax[0].tick_params(axis='y', labelsize=ticklabelsize)
        ax[0].fill_between(k, ps_gen-ps_gen_iqr, ps_gen+ps_gen_iqr,alpha=0.3, facecolor=gen_gt_color)
        ax[0].fill_between(k, ps_ip-ps_ip_iqr, ps_ip+ps_ip_iqr,alpha=0.1, facecolor=ip_gt_color)
        ax[0].fill_between(k, ps_gt-ps_gt_iqr, ps_gt+ps_gt_iqr,alpha=0.1, facecolor='black')
        ax[0].set_ylabel('$P(k)$', fontsize=titlesize)

        ax[1].set_xscale('log')
        ax[1].plot(k, 100 * (ps_gen_median - ps_gt_median) / ps_gt_median, c=gen_gt_color)
        ax[1].plot(k, 100 * (ps_ip_median - ps_gt_median) / ps_gt_median, c=ip_gt_color)
        ax[1].axhline(y=0, c='black', linestyle='--')
        ax[1].set_ylabel(r'$\dfrac{P(k)}{P_{f(R)}(k)} - 1$ (%)', fontsize=axeslabelsize)
        ax[1].set_ylabel('(P(k)/P_fr(k)) - 1')
        ax[1].set_xlabel('$k$ (h/Mpc)', fontsize=axeslabelsize);
        ax[1].tick_params(axis='x', labelsize=ticklabelsize)
        ax[1].tick_params(axis='y', labelsize=ticklabelsize)
        ax[1].fill_between(k, -25, 25, alpha=0.2)
        ax[1].fill_between(k, -ps_gt_std, ps_gt_std, alpha=0.1)
        ax[0].set_xlim([k.min(), 5])
        ax[1].set_xlim([k.min(), 5])
        ax[1].set_ylim([-100, 100])
        ax[1].set_yticks(np.arange(-100, +125, 25))
        plt.savefig(f'FIGURES/{field_name}/ps_{name}_median_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)

        np.save(f'FIGURES/{field_name}/ps_ip_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_ip_median)
        np.save(f'FIGURES/{field_name}/ps_gen_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gen_median)
        np.save(f'FIGURES/{field_name}/ps_gt_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gt_median)
        np.save(f'FIGURES/{field_name}/k_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', k)
        np.save(f'FIGURES/{field_name}/iqr_ip_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_ip_iqr)
        np.save(f'FIGURES/{field_name}/iqr_gen_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gen_iqr)
        np.save(f'FIGURES/{field_name}/iqr_gt_{name}_median_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gt_iqr)

        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].loglog(k, ps_gen, c=gen_gt_color, label=fr'cGAN: Dist = {chisquare_gen_gt:.2f}')
        ax[0].loglog(k, ps_ip, c=ip_gt_color, label=fr'GR sim: Dist = {chisquare_ip_gt:.2f}')
        ax[0].loglog(k, ps_gt, c='black', label='f(R) sim')
        ax[0].legend()
        ax[0].set_title('Mean power spectrum', fontsize=titlesize)
        ax[0].tick_params(axis='x', labelsize=ticklabelsize)
        ax[0].tick_params(axis='y', labelsize=ticklabelsize)
        ax[0].fill_between(k, ps_gen-ps_gen_std, ps_gen+ps_gen_std ,alpha=0.3, facecolor=gen_gt_color)
        ax[0].fill_between(k, ps_ip-ps_ip_std, ps_ip+ps_ip_std ,alpha=0.1, facecolor=ip_gt_color)
        ax[0].fill_between(k, ps_gt-ps_gt_std, ps_gt+ps_gt_std ,alpha=0.1, facecolor='black')
        ax[0].set_ylabel('$P(k)$', fontsize=titlesize)

        ax[1].set_xscale('log')
        ax[1].plot(k, 100 * (ps_gen - ps_gt) / ps_gt, c=gen_gt_color)
        ax[1].plot(k, 100 * (ps_ip - ps_gt) / ps_gt, c=ip_gt_color)
        ax[1].axhline(y=0, c='black', linestyle='--')
        ax[1].set_ylabel(r'$\dfrac{P(k)}{P_{f(R)}(k)} - 1$ (%)', fontsize=axeslabelsize)
        ax[1].set_ylabel('(P(k)/P_fr(k)) - 1')
        ax[1].set_xlabel('$k$ (h/Mpc)', fontsize=axeslabelsize);
        ax[1].tick_params(axis='x', labelsize=ticklabelsize)
        ax[1].tick_params(axis='y', labelsize=ticklabelsize)
        ax[1].fill_between(k, -25, 25, alpha=0.2)
        ax[1].fill_between(k, -ps_gt_std, ps_gt_std, alpha=0.1)
        ax[0].set_xlim([k.min(), 5])
        ax[1].set_xlim([k.min(), 5])
        ax[1].set_ylim([-100, 100])
        ax[1].set_yticks(np.arange(-100, +125, 25))
        plt.savefig(f'FIGURES/{field_name}/ps_{name}_mean_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)

        np.save(f'FIGURES/{field_name}/ps_ip_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_ip)
        np.save(f'FIGURES/{field_name}/ps_gen_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gen)
        np.save(f'FIGURES/{field_name}/ps_gt_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gt)
        np.save(f'FIGURES/{field_name}/k_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', k)
        np.save(f'FIGURES/{field_name}/iqr_ip_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_ip_std)
        np.save(f'FIGURES/{field_name}/iqr_gen_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gen_std)
        np.save(f'FIGURES/{field_name}/iqr_gt_{name}_mean_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ps_gt_std)

    """
    #### Repeat the relative difference plot as above but taking input GR as reference ####
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(hspace=0)

    ax.set_xscale('log')
    ax.plot(k, 100 * (ps_gen - ps_ip) / ps_ip, c=ip_gen_color, label='cGAN generated')
    ax.plot(k, 100 * (ps_gt - ps_ip) / ps_ip, c=ip_gt_color, label='f(R) simulation')
    ax.axhline(y=0, c='black', linestyle='--')
    ax.set_ylabel('Relative difference (%)', fontsize=14)
    ax.set_xlabel('k (h/Mpc)', fontsize=14);
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.fill_between(k, -25, 25, alpha=0.2)
    ax.set_xlim([k.min(), 15.])
    ax.set_xlim([k.min(), 15.])
    ax.set_ylim([-100, 100])
    ax.set_title('Reference is GR simulation')
    ax.legend()
    ax.set_yticks(np.arange(-100, +125, 25))

    #######################################################################################
    """
    """
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
    """

    #del corr_gen_gt, corr_ip_gt, ps_gen, ps_ip, ps_gt
    #gc.collect()

    # 2. CUMULANTS
    
    gens_cumulant_x, gens_cumulant_y, _, gens_cumulant_y_std, gens_cumulant_y_median, gens_cumulant_y_iqr, gens_cumulant_x_median, _ = cumulant_overall(gens, n=2)
    ips_cumulant_x, ips_cumulant_y, _, ips_cumulant_y_std, ips_cumulant_y_median, ips_cumulant_y_iqr, ips_cumulant_x_median, _ = cumulant_overall(ips, n=2)
    gts_cumulant_x, gts_cumulant_y, _, gts_cumulant_y_std, gts_cumulant_y_median, gts_cumulant_y_iqr, gts_cumulant_x_median, gts_cumulant_x_iqr = cumulant_overall(gts, n=2)

    print("Distance for cumulants (n = 2).....")
    chisquare_ip_gt_cumulant = ps_metric(10**ips_cumulant_y, 10**gts_cumulant_y).mean()
    chisquare_gen_gt_cumulant = ps_metric(10**gens_cumulant_y, 10**gts_cumulant_y).mean()
    chisquare_ip_gt_cumulant_median = ps_metric(10**ips_cumulant_y_median, 10**gts_cumulant_y_median).mean()
    chisquare_gen_gt_cumulant_median = ps_metric(10**gens_cumulant_y_median, 10**gts_cumulant_y_median).mean()

    print(f'Distance between mean cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant}')
    print(f'Distance between median cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant_median}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant_median}')

    if save:
        np.save(f'FIGURES/{field_name}/cumulant2_y_gen_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y)
        np.save(f'FIGURES/{field_name}/cumulant2_y_ip_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gt_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gen_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_median)
        np.save(f'FIGURES/{field_name}/cumulant2_y_ip_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_median)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_median)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gen_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_std)
        np.save(f'FIGURES/{field_name}/cumulant2_y_ip_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_std)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gt_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_std)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gen_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_iqr)
        np.save(f'FIGURES/{field_name}/cumulant2_y_ip_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_iqr)
        np.save(f'FIGURES/{field_name}/cumulant2_y_gt_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_iqr)
        np.save(f'FIGURES/{field_name}/cumulant2_x_gt_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x)
        np.save(f'FIGURES/{field_name}/cumulant2_x_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x_median)

    gens_cumulant_x3, gens_cumulant_y3, _, gens_cumulant_y_std3, gens_cumulant_y_median3, gens_cumulant_y_iqr3, gens_cumulant_x_median3, _ = cumulant_overall(gens, n=3)
    ips_cumulant_x3, ips_cumulant_y3, _, ips_cumulant_y_std3, ips_cumulant_y_median3, ips_cumulant_y_iqr3, ips_cumulant_x_median3, _ = cumulant_overall(ips, n=3)
    gts_cumulant_x3, gts_cumulant_y3, _, gts_cumulant_y_std3, gts_cumulant_y_median3, gts_cumulant_y_iqr3, gts_cumulant_x_median3, gts_cumulant_x_iqr3 = cumulant_overall(gts, n=3)

    print("Distance for cumulants (n = 3).....")
    chisquare_ip_gt_cumulant = ps_metric(10**ips_cumulant_y3, 10**gts_cumulant_y3).mean()
    chisquare_gen_gt_cumulant = ps_metric(10**gens_cumulant_y3, 10**gts_cumulant_y3).mean()
    chisquare_ip_gt_cumulant_median = ps_metric(10**ips_cumulant_y_median3, 10**gts_cumulant_y_median3).mean()
    chisquare_gen_gt_cumulant_median = ps_metric(10**gens_cumulant_y_median3, 10**gts_cumulant_y_median3).mean()

    print(f'Distance between mean cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant}')
    print(f'Distance between median cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant_median}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant_median}')

    if save:
        np.save(f'FIGURES/{field_name}/cumulant3_y_gen_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_ip_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gt_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gen_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_median3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_ip_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_median3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_median3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gen_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_std3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_ip_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_std3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gt_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_std3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gen_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_iqr3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_ip_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_iqr3)
        np.save(f'FIGURES/{field_name}/cumulant3_y_gt_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_iqr3)
        np.save(f'FIGURES/{field_name}/cumulant3_x_gt_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x3)
        np.save(f'FIGURES/{field_name}/cumulant3_x_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x_median3)

    gens_cumulant_x4, gens_cumulant_y4, _, gens_cumulant_y_std4, gens_cumulant_y_median4, gens_cumulant_y_iqr4, gens_cumulant_x_median4, _ = cumulant_overall(gens, n=4)
    ips_cumulant_x4, ips_cumulant_y4, _, ips_cumulant_y_std4, ips_cumulant_y_median4, ips_cumulant_y_iqr4, ips_cumulant_x_median4, _ = cumulant_overall(ips, n=4)
    gts_cumulant_x4, gts_cumulant_y4, _, gts_cumulant_y_std4, gts_cumulant_y_median4, gts_cumulant_y_iqr4, gts_cumulant_x_median4, gts_cumulant_x_iqr4 = cumulant_overall(gts, n=4)

    print("Distance for cumulants (n = 4).....")
    chisquare_ip_gt_cumulant = ps_metric(10**ips_cumulant_y4, 10**gts_cumulant_y4).mean()
    chisquare_gen_gt_cumulant = ps_metric(10**gens_cumulant_y4, 10**gts_cumulant_y4).mean()
    chisquare_ip_gt_cumulant_median = ps_metric(10**ips_cumulant_y_median4, 10**gts_cumulant_y_median4).mean()
    chisquare_gen_gt_cumulant_median = ps_metric(10**gens_cumulant_y_median4, 10**gts_cumulant_y_median4).mean()

    print(f'Distance between mean cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant}')
    print(f'Distance between median cumulants:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_cumulant_median}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_cumulant_median}')

    if save:
        np.save(f'FIGURES/{field_name}/cumulant4_y_gen_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_ip_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gt_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gen_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_median4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_ip_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_median4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_median4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gen_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_std4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_ip_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_std4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gt_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_std4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gen_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gens_cumulant_y_iqr4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_ip_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', ips_cumulant_y_iqr4)
        np.save(f'FIGURES/{field_name}/cumulant4_y_gt_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_y_iqr4)
        np.save(f'FIGURES/{field_name}/cumulant4_x_gt_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x4)
        np.save(f'FIGURES/{field_name}/cumulant4_x_gt_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', gts_cumulant_x_median4)

    assert np.all(gens_cumulant_x == ips_cumulant_x)
    assert np.all(ips_cumulant_x == gts_cumulant_x)

    if save:
        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x, ips_cumulant_y, label='GR sim')
        ax[0].plot(gts_cumulant_x, gts_cumulant_y, label='f(R) sim')
        ax[0].plot(gens_cumulant_x, gens_cumulant_y, label='cGAN')
        ax[0].set_ylabel(r'$\log{\sigma^2}$')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x, ((10**ips_cumulant_y)/(10**gts_cumulant_y)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x, ((10**gens_cumulant_y)/(10**gts_cumulant_y)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta\sigma^2$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_variance_{name}_mean_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)


        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x3, ips_cumulant_y3, label='GR sim')
        ax[0].plot(gts_cumulant_x3, gts_cumulant_y3, label='f(R) sim')
        ax[0].plot(gens_cumulant_x3, gens_cumulant_y3, label='cGAN')
        ax[0].set_ylabel('Skewness')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x3, ((10**ips_cumulant_y3)/(10**gts_cumulant_y3)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x3, ((10**gens_cumulant_y3)/(10**gts_cumulant_y3)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta Skewness$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_skewness_{name}_mean_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)


        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x4, ips_cumulant_y4, label='GR sim')
        ax[0].plot(gts_cumulant_x4, gts_cumulant_y4, label='f(R) sim')
        ax[0].plot(gens_cumulant_x4, gens_cumulant_y4, label='cGAN')
        ax[0].set_ylabel('Kurtosis')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x4, ((10**ips_cumulant_y4)/(10**gts_cumulant_y4)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x4, ((10**gens_cumulant_y4)/(10**gts_cumulant_y4)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta Kurtosis$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_kurtosis_{name}_mean_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)


        # Now median cumulant plots.
        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x_median, ips_cumulant_y_median, label='GR sim')
        ax[0].plot(gts_cumulant_x_median, gts_cumulant_y_median, label='f(R) sim')
        ax[0].plot(gens_cumulant_x_median, gens_cumulant_y_median, label='cGAN')
        ax[0].set_ylabel(r'$\log{\sigma^2}$')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x_median, ((10**ips_cumulant_y_median)/(10**gts_cumulant_y_median)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x_median, ((10**gens_cumulant_y_median)/(10**gts_cumulant_y_median)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta\sigma^2$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_variance_{name}_median_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)


        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x_median3, ips_cumulant_y_median3, label='GR sim')
        ax[0].plot(gts_cumulant_x_median3, gts_cumulant_y_median3, label='f(R) sim')
        ax[0].plot(gens_cumulant_x_median3, gens_cumulant_y_median3, label='cGAN')
        ax[0].set_ylabel('Skewness')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x_median3, ((10**ips_cumulant_y_median3)/(10**gts_cumulant_y_median3)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x_median3, ((10**gens_cumulant_y_median3)/(10**gts_cumulant_y_median3)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta Skewness$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_skewness_{name}_median_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)


        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0)

        ax[0].plot(ips_cumulant_x_median4, ips_cumulant_y_median4, label='GR sim')
        ax[0].plot(gts_cumulant_x_median4, gts_cumulant_y_median4, label='f(R) sim')
        ax[0].plot(gens_cumulant_x_median4, gens_cumulant_y_median4, label='cGAN')
        ax[0].set_ylabel('Kurtosis')
        ax[0].set_xlabel(r'$\log{R_{th}}$')

        ax[0].set_ylim(bottom=0.5)
        ax[0].legend()

        ax[1].plot(ips_cumulant_x_median4, ((10**ips_cumulant_y_median4)/(10**gts_cumulant_y_median4)) - 1, label='GR sim <-> f(R) sim')
        ax[1].plot(gens_cumulant_x_median4, ((10**gens_cumulant_y_median4)/(10**gts_cumulant_y_median4)) - 1, label='cGAN <-> f(R) sim')
        # ax[1].plot(x, (yf4-ygr), label='f4')
        # ax[1].plot(x, (yf5-ygr), label='f5')
        # ax[1].plot(x, (yf6-ygr), label='f6')
        ax[1].set_ylim([-0.05, 0.5])
        #ax[1].axhline(y=0, linestyle='--', c='black')
        ax[1].legend()
        ax[1].set_xlabel(r'$\log{R_{th}}$')
        ax[1].set_ylabel(r'$\Delta Kurtosis$')

        plt.savefig(f'FIGURES/{field_name}/cumulant_kurtosis_{name}_median_{epoch_num}epoch_{val_or_test}.png', bbox_inches='tight', dpi=250)



    # 3. PIXEL DISTANCE
    print('y_real PDF')
    x_real, y_real, _, y_real_std, y_real_median, y_real_iqr, x_real_median, x_real_iqr = pixel_pdf(gts)
    print('y_fake PDF')
    x_fake, y_fake, _, y_fake_std, y_fake_median, y_fake_iqr, _, _ = pixel_pdf(gens)
    print('y_ip PDF')
    x_ip, y_ip, _, y_ip_std, y_ip_median, y_ip_iqr, _, _ = pixel_pdf(ips)

    assert np.all(x_real == x_fake)
    assert np.all(x_real == x_ip)

    chisquare_ip_gt_PDF =  wasserstein_distance_norm(y_real, y_ip)
    chisquare_gen_gt_PDF = wasserstein_distance_norm(y_real, y_fake)

    print(f'1-Wasserstein distance between mean PDF:\n\tbetween ground-truth f(R) and input GR: {chisquare_ip_gt_PDF}\n\tbetween ground-truth f(R) and generated f(R): {chisquare_gen_gt_PDF}')

    if save:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.plot(x_real, y_fake, label=f'cGAN: Chi: {chisquare_gen_gt_PDF:.4f}', c=gen_gt_color, alpha=0.7)
        ax.plot(x_real, y_real, label=f'F4 sim', c='black', alpha=0.7)
        ax.plot(x_real, y_ip, label=f'GR sim: Chi: {chisquare_ip_gt_PDF:.4f}', c=ip_gt_color, alpha=0.7)
        ax.tick_params(axis='x', labelsize=ticklabelsize)
        ax.tick_params(axis='y', labelsize=ticklabelsize)
        ax.legend(fontsize=ticklabelsize)
        ax.set_xscale('log');
        ax.set_yscale('log')
        ax.set_ylabel('Pixel count', fontsize=axeslabelsize)
        ax.set_xlabel('Pixel density value', fontsize=axeslabelsize)
        plt.savefig(f'FIGURES/{field_name}/mass_hist_{name}_{epoch_num}epoch_f4_den_{val_or_test}.png', bbox_inches='tight', dpi=250)

        np.save(f'FIGURES/{field_name}/mass_ip_hist_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_ip)
        np.save(f'FIGURES/{field_name}/mass_gen_hist_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_fake)
        np.save(f'FIGURES/{field_name}/mass_gt_hist_mean_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_real)
        np.save(f'FIGURES/{field_name}/mass_ip_hist_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_ip_median)
        np.save(f'FIGURES/{field_name}/mass_gen_hist_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_fake_median)
        np.save(f'FIGURES/{field_name}/mass_gt_hist_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_real_median)
        np.save(f'FIGURES/{field_name}/mass_ip_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_ip_std)
        np.save(f'FIGURES/{field_name}/mass_gen_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_fake_std)
        np.save(f'FIGURES/{field_name}/mass_gt_std_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_real_std)
        np.save(f'FIGURES/{field_name}/mass_ip_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_ip_iqr)
        np.save(f'FIGURES/{field_name}/mass_gen_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_fake_iqr)
        np.save(f'FIGURES/{field_name}/mass_gt_iqr_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', y_real_iqr)
        np.save(f'FIGURES/{field_name}/mass_hist_xaxis_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', x_real)
        np.save(f'FIGURES/{field_name}/mass_hist_xaxis_median_{name}_{epoch_num}epoch_{field_name}_{val_or_test}.npy', x_real_median)

    return chisquare_ip_gt_median, chisquare_gen_gt_median, chisquare_ip_gt, chisquare_gen_gt, chisquare_ip_gt_PDF, chisquare_gen_gt_PDF

    ################################ Performing two-sample ks test on pixel distribution ################################
    print("Between generated and ground-truth f(R)")
    print(ks_2samp(gens.flatten(), gts.flatten()))
    #####################################################################################################################
    """
    # return chisquare_ip_gt_median, chisquare_gen_gt_median, chisquare_ip_gt, chisquare_gen_gt, wass_pixel_gt_ip, wass_pixel_gt_gen
    return chisquare_ip_gt_median, chisquare_gen_gt_median, chisquare_ip_gt, chisquare_gen_gt
    #return

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
    """
