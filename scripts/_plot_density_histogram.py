import glob
import numpy as np
import matplotlib.pyplot as plt

from pix2pix_modified_gravity.evaluation_metrics import pixel_pdf

def get_and_plot_pixel_pdf(arrs, bins=np.logspace(start=-2, stop=2, num=99), name='image.png'):
    x, y, xstd, ystd, ymedian, ymad, xmedian, xmad = pixel_pdf(arrs)

    linewidth, ticklabelsize, axeslabelsize, titlesize = 2, 14, 14, 16
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))
    fig.subplots_adjust(hspace=0)
    ax.plot(x, y, label='PDF', c='black', alpha=0.7, linewidth=linewidth)
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize)
    ax.legend(fontsize=ticklabelsize)
    ax.set_xscale('log');
    ax.set_yscale('log')
    ax.set_ylabel('Counts', fontsize=axeslabelsize)
    ax.set_title('Density histogram', fontsize=titlesize)
    # ax[0].fill_between(x, y_gen-y_gen_std, y_gen+y_gen_std, alpha=0.2, facecolor=gen_gt_color)

    """
    ax[1].plot(xgt, 100 * ((y_ip - y_gt) / y_gt), c=ip_gt_color, linewidth=linewidth)
    ax[1].plot(xgt, 100 * ((y_gen - y_gt) / y_gt), c=gen_gt_color, linewidth=linewidth)
    ax[1].set_ylim([-15, +15])
    ax[1].axhline(y=0, c='black', linestyle='--', linewidth=linewidth)
    ax[1].tick_params(axis='x', labelsize=ticklabelsize)
    ax[1].tick_params(axis='y', labelsize=ticklabelsize)
    ax[1].set_xlabel(r'$\rho / \rho_{mean}$', fontsize=axeslabelsize)
    ax[1].set_ylabel(r'$\dfrac{Counts}{Counts_{F4}} - 1$ (%)', fontsize=axeslabelsize)
    ax[1].set_yticks(np.arange(-15, +20, 5))
    ax[1].fill_between(xgt, -5, 5, alpha=0.2)
    """
    plt.savefig(name)
    plt.show()

bins = np.logspace(start=-2, stop=4, num=99)

arrs, ratio = [], []
for img in glob.glob('official_pix2pix_data_F4n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])
    ratio.append(arr[:, 512:]/arr[:, :512])

arrs = np.array(arrs)
ratio = np.array(ratio)

get_and_plot_pixel_pdf(arrs, name='GR_original.png')
get_and_plot_pixel_pdf(ratio, bins=np.logspace(-2, 2, 50), name='Ratio_original.png')



import sys
sys.exit()





arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_GR.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F4.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F5.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F6.png', bbox_inches='tight', dpi=200)
plt.close()

