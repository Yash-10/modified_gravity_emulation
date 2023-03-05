#############################################
# Before running, do:

#```
#module unload python/3.6.5
#module load python/3.8.7-C8
#```
#############################################
import os
import importlib
from functools import partial

#os.system('pip install numpy==1.21')
os.chdir("/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix")

from pix2pix_modified_gravity.evaluation_metrics import (
    ps_2d,
    wasserstein_distance_norm,
    peak_count,
    mssim_single,
    mean_density,
    median_density,
    correlation_coefficient,
    driver
)  # Import all functions.

def andres_forward(x, shift=20., scale=1.):
    """Map real positive numbers to a [-scale, scale] range.
    Numpy version
    """
    return scale * (2 * (x / (x + 1 + shift)) - 1)

def andres_backward(y, shift=20., scale=1., real_max=1e8):
    """Inverse of the function forward map.
    Numpy version
    """
    simple_max = andres_forward(real_max, shift, scale)
    simple_min = andres_forward(0, shift, scale)
    y_clipped = np.clip(y, simple_min, simple_max) / scale
    return (shift + 1) * (y_clipped + 1) / (1 - y_clipped)

import os
import subprocess
import numpy as np
import glob

#for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
for i in [110]:
    print(f"------------------------- {i} MODEL CHECK ----------------------------")

    subprocess.call(['cp', f'/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/pix2pix_F6n1_GR_VELDIV/{i}_net_G.pth', '/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/pix2pix_F6n1_GR_VELDIV/latest_net_G.pth'])

    subprocess.call(['python3', 'test.py', '--dataroot', '/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F6n1_GR', '--name', 'pix2pix_F6n1_GR_VELDIV', '--model', 'pix2pix', '--direction', 'AtoB', '--batch_size', '1', '--input_nc', '1', '--output_nc', '1', '--num_threads=2', '--norm', 'instance', '--load_size', '512', '--crop_size', '512', '--no_flip', '--eval', '--phase', 'test', '--num_test', '461', '--results_dir', 'test_results', '--netG', 'unet_512', '--ndf', '64', '--ngf', '128', '--no_dropout', '--gpu_ids', '-1'])    

    FIELD_TYPE = 'veldiv'

    result_imgs = sorted(glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/test_results/pix2pix_F6n1_GR_VELDIV/test_latest/images/*npy'))
    assert len(result_imgs) != 0
    for i in range(0, len(result_imgs), 3):
        assert "fake_B" in result_imgs[i]
        assert "real_A" in result_imgs[i+1]
        assert "real_B" in result_imgs[i+2]
        gr = np.load(result_imgs[i+1]).squeeze()
        fr = np.load(result_imgs[i+2]).squeeze()
        #print(f'{result_imgs[i+1].split("/")[-1].split("_real")[0]}.npy')
        test_curr = np.load(
                os.path.join('/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F6n1_GR/test', f'{result_imgs[i+1].split("/")[-1].split("_real")[0]}.npy')
        )
        orig_gr_test = test_curr[:, :512]
        orig_fr_test = test_curr[:, 512:]
       
        if FIELD_TYPE != 'veldiv':
            assert np.allclose(gr, orig_gr_test, atol=1e-1)
            assert np.allclose(fr, orig_fr_test, atol=1e-1)

    import glob
    import numpy as np
    import matplotlib.pyplot as plt


    """
    print("===== First showing a single example =====")
    results_list = sorted(glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/val_results/pix2pix_F6n1_GR_new/val_latest/images/*npy'))
    for i in range(0, len(results_list), 3):
        assert "fake_B" in results_list[i]
        assert "real_A" in results_list[i+1]
        assert "real_B" in results_list[i+2]

        gen = np.load(results_list[i]).squeeze()
        gr = np.load(results_list[i+1]).squeeze()
        fr = np.load(results_list[i+2]).squeeze()

        fig, ax = plt.subplots(1, 3, figsize=(15, 7))
        if FIELD_TYPE == 'den':
            fig.suptitle('Density fields are shown transformed; however power spectra and all other stats are on untransformed ones')
            ax[0].imshow(andres_forward(gr, scale=1., shift=4.))  # Change shift values for different experiments
            ax[0].set_title("GR (input)")
            ax[1].imshow(andres_forward(fr, scale=1., shift=4.))
            ax[1].set_title("f(R) (ground_truth)")
            ax[2].imshow(andres_forward(gen, scale=1., shift=4.))
            ax[2].set_title("f(R) (generated)")
            plt.show()
        elif FIELD_TYPE == 'veldiv':
            fig.suptitle('Velocity divergence fields are shown transformed; however power spectra and all other stats are on untransformed ones')
            ax[0].imshow(veldiv_forward(gr, scale=1., shift=4.))  # Change shift values for different experiments
            ax[0].set_title("GR (input)")
            ax[1].imshow(veldiv_forward(fr, scale=1., shift=4.))
            ax[1].set_title("f(R) (ground_truth)")
            ax[2].imshow(veldiv_forward(gen, scale=1., shift=4.))
            ax[2].set_title("f(R) (generated)")
            plt.show()

        # ** Note ** images are shown transformed but power spectrum calculated on the untransformed images.

        kgen, Pkgen = ps_2d(gen)
        kgr, Pkgr = ps_2d(gr)
        kfr, Pkfr = ps_2d(fr)

        fig, ax = plt.subplots(1, 1)
        ax.loglog(kgr, Pkgr, c='red', label='gr')
        ax.loglog(kgen, Pkgen, c='green', label='gen')
        ax.loglog(kfr, Pkfr, c='blue', label='fr')
        ax.legend()
        plt.show()
        break
    """

    print("===== Showing results across multiple images =====")
    #### Validate on all validation images ####
    results_list = sorted(glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/test_results/pix2pix_F6n1_GR_VELDIV/test_latest/images/*npy'))
    assert len(results_list) != 0
    val_gen = np.empty((int(len(results_list)/3), 512, 512))
    val_ip = np.empty((int(len(results_list)/3), 512, 512))
    val_gt = np.empty((int(len(results_list)/3), 512, 512))
    counter = 0
    for i in range(0, len(results_list), 3):
        assert "fake_B" in results_list[i]
        assert "real_A" in results_list[i+1]
        assert "real_B" in results_list[i+2]

        gen = np.load(results_list[i]).squeeze()
        gr = np.load(results_list[i+1]).squeeze()
        fr = np.load(results_list[i+2]).squeeze()

        #print(gr)
        #print(fr)
        #print(gen)

        val_gen[counter] = gen
        val_ip[counter] = gr
        val_gt[counter] = fr

        counter += 1

    assert counter == int(len(results_list)/3)

    driver(val_gen, val_ip, val_gt, vel_field=True)

