######################################################################################################################################
###### The functions in this script have been taken from https://github.com/nperraud/gantools/blob/master/gantools/plot/plot.py ######
######################################################################################################################################

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy

import os
import warnings

def plot_histogram(x, histo, yscale='log', tick_every=10, bar_width=1):
    plt.bar(np.arange(len(histo)), histo, bar_width)
    positions, labels = ([], [])
    for idx in range(len(x)):
        if idx == len(x) - 1:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
        if idx % tick_every == 0:
            positions.append(idx)
            labels.append(np.round(x[idx], 2))
    plt.xticks(positions, labels)
    plt.yscale(yscale)

    

# Comparison plot between two curves real and fake corresponding to axis x
def plot_cmp(x, fake, real=None, xscale='linear', yscale='log', xlabel="", ylabel="", ax=None, title="", shade=False, confidence=None, xlim=None, ylim=None, fractional_difference=False, algorithm='classic', loc=3):
    if ax is None:
        ax = plt.gca()
    if real is None:
        # Plot a line that will be canceled by fake
        # In this way the fractional difference is zero
        plot_single(x, fake, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=False, confidence=confidence, xlim=xlim, ylim=ylim)
    else:
        plot_single(x, real, color='b', ax=ax, label="Real", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    plot_single(x, fake, color='r', ax=ax, label="Fake", xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel, title=title, shade=shade, confidence=confidence, xlim=xlim, ylim=ylim)
    ax.legend(fontsize=14, loc=loc)
    if fractional_difference:
        if real is None:
            real = fake
        if shade:
            real = np.mean(real, axis=0)
            fake = np.mean(fake, axis=0)
        plot_fractional_difference(x, real, fake, xscale=xscale, yscale='linear', ax=ax, color='g', ylim=ylim[1] if isinstance(ylim, list) else None, algorithm=algorithm, loc=loc)
        
# Plot an image
def plot_img(img, x=None, title="", ax=None, cmap=plt.cm.plasma, vmin=None, vmax=None, tick_every=10, colorbar=False, log_norm=False, kwargs_imshow=None):
    if kwargs_imshow is None:
        kwargs_imshow = {}
    if ax is None:
        ax = plt.gca()
    img = np.squeeze(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, norm=LogNorm() if log_norm else None, **kwargs_imshow)
    ax.set_title(title)
    ax.title.set_fontsize(16)
    if x is not None:
        # Define axes
        ticklabels = []
        for i in range(len(x)):
            if i % tick_every == 0:
                ticklabels.append(str(int(round(x[i], 0))))
        ticks = np.linspace(0, len(x) - (len(x) % tick_every), len(ticklabels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
    else:
        ax.axis('off')
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im


# Plot fractional difference
# if algorithm='classic' then the classic fractional difference abs(real - fake) / real is computed
# if algorithm='relative' then abs(real-fake) / std(real) is computed
def plot_fractional_difference(x, real, fake, xlim=None, ylim=None, xscale="log", yscale="linear", title="", ax=None, color='b', algorithm='classic', loc=3):
    if ax is None:
        ax1 = plt.gca()
    else:
        ax1 = ax.twinx()
    diff = np.abs(real - fake)
    if algorithm == 'classic':
        diff = diff / real
    elif algorithm == 'relative':
        diff = diff / np.std(real, axis=0)
    else:
        raise ValueError("Unknown algorithm name " + str(algorithm))
    plot_single(x, diff, ax=ax1, xscale=xscale, yscale=yscale, title=title, color=color, label='Frac. diff.' if algorithm == 'classic' else 'Rel. diff.', ylim=ylim)
    ax1.tick_params(axis='y', labelsize=12, labelcolor=color)

    # Adjust legends
    if ax is not None:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax1.get_legend_handles_labels()
        ax.get_legend().remove()
        ax1.legend(lines + lines2, labels + labels2, loc=loc, fontsize=14)


# Plot a line with a shade representing either the confidence interval or the standard error
# If confidence is None then the standard error is shown
def plot_with_shade(ax, x, y, label, color, confidence=None, **linestyle):
    transparency = 0.2

    n = y.shape[0]
    y_mean = np.mean(y, axis=0)
    error = (np.var(y, axis=0) / n)**0.5
    if confidence == 'std':
        error = np.std(y, axis=0)
    elif confidence is not None:
        error = error * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    ax.plot(x, y_mean, label=label, color=color, **linestyle)
    ax.fill_between(
        x, y_mean - error, y_mean + error, alpha=transparency, color=color)

# Plot a single curve
def plot_single(x, y, color='b', ax=None, label=None, xscale='linear', yscale='log', xlabel="", ylabel="", title="", shade=False, confidence=None, xlim=None, ylim=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    linestyle = {
        "linewidth": 1,
        "markeredgewidth": 0,
        "markersize": 3,
        "marker": "o",
        "linestyle": "-"
    }
    if shade:
        plot_with_shade(ax, x, y, label=label, color=color, confidence=confidence, **linestyle)
    else:
        if confidence is not None:
            y = np.mean(y, axis=0)
        ax.plot(x, y, label=label, color=color, **linestyle)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        if isinstance(ylim, list):
            ylim = ylim[0]
        ax.set_ylim(ylim)
    ax.title.set_text(title + "\n")
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if label:
        ax.legend(fontsize=14, loc=3)
