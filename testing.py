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

from evaluation_metrics import (
    ps_2d,
    wasserstein_distance_norm,
    peak_count,
    mssim_single,
    mean_density,
    median_density,
    correlation_coefficient,
    driver
)  # Import all functions.
#basedataset = importlib.import_module('pix2pix_modified_gravity.pytorch-CycleGAN-and-pix2pix.data.base_dataset')
#os.chdir('pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix')
#base = importlib.import_module('pix2pix_modified_gravity.pytorch-CycleGAN-and-pix2pix.data.base_dataset')
from util.util import andres_backward
from data.base_dataset import andres_forward

os.system("cp /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/pix2pix_F6n1_GR_new/10_net_G.pth /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/latest_net_G.pth")

os.system("python3 test.py --dataroot /cosma5/data/durham/dc-gond1/official_pix2pix_data_F5n1andF6n1_traindata_combined_GR --name pix2pix_F6n1_GR_new \
                         --model pix2pix --direction AtoB --batch_size 1 --input_nc 1 --output_nc 1 --num_threads=2 --norm instance \
                                          --load_size 512 --crop_size 512 --no_flip --eval --phase val --num_test 461 --results_dir val_results --netG unet_512 \
                                                           --ndf 64 --ngf 128 --no_dropout --gpu_ids -1")
import os
import numpy as np
import glob

result_imgs = sorted(glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/val_results/pix2pix_F6n1_GR_new/val_latest/images/*npy'))
assert len(result_imgs) != 0
for i in range(0, len(result_imgs), 3):
    assert "fake_B" in result_imgs[i]
    assert "real_A" in result_imgs[i+1]
    assert "real_B" in result_imgs[i+2]
    gr = np.load(result_imgs[i+1]).squeeze()
    fr = np.load(result_imgs[i+2]).squeeze()
    print(f'{result_imgs[i+1].split("/")[-1].split("_real")[0]}.npy')
    test_curr = np.load(
            os.path.join('/cosma5/data/durham/dc-gond1/official_pix2pix_data_F5n1andF6n1_traindata_combined_GR/val', f'{result_imgs[i+1].split("/")[-1].split("_real")[0]}.npy')
    )
    orig_gr_test = test_curr[:, :512]
    orig_fr_test = test_curr[:, 512:]
    print(orig_gr_test)
    assert np.allclose(gr, orig_gr_test, rtol=1e-2)
    assert np.allclose(fr, orig_fr_test, rtol=1e-2)

import glob
import numpy as np
import matplotlib.pyplot as plt

FIELD_TYPE = 'den'

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
        ax[0].imshow(base.andres_forward(gr, scale=1., shift=4.))  # Change shift values for different experiments
        ax[0].set_title("GR (input)")
        ax[1].imshow(base.andres_forward(fr, scale=1., shift=4.))
        ax[1].set_title("f(R) (ground_truth)")
        ax[2].imshow(base.andres_forward(gen, scale=1., shift=4.))
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

print("===== Showing results across multiple images =====")
#### Validate on all validation images ####
results_list = sorted(glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/val_results/pix2pix_F6n1_GR_new/val_latest/images/*npy'))
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

    val_gen[counter] = gen
    val_ip[counter] = gr
    val_gt[counter] = fr

    counter += 1

assert counter == int(len(results_list)/3)

driver(val_gen, val_ip, val_gt)  # Only check first 100 otherwise RAM blows up.?
