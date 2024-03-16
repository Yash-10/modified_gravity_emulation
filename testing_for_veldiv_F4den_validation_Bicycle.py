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
os.chdir("/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN")

from pix2pix_modified_gravity.evaluation_metrics import (
#    ps_2d,
#    wasserstein_distance_norm,
#    peak_count,
#    mssim_single,
#    mean_density,
#    median_density,
#    correlation_coefficient,
    driver
)  # Import all functions.

#def andres_forward(x, shift=20., scale=1.):
#    """Map real positive numbers to a [-scale, scale] range.
#    Numpy version
#    """
#    return scale * (2 * (x / (x + 1 + shift)) - 1)
#
#def andres_backward(y, shift=20., scale=1., real_max=1e8):
##    """Inverse of the function forward map.
#    Numpy version
#    """
#    simple_max = andres_forward(real_max, shift, scale)
#    simple_min = andres_forward(0, shift, scale)
#    y_clipped = np.clip(y, simple_min, simple_max) / scale
#    return (shift + 1) * (y_clipped + 1) / (1 - y_clipped)

FIELD_TYPE = 'veldiv'
import os
import subprocess
import numpy as np
import glob
from test_utils import hvstack, grouper

name = 'CHECK_F4_veldiv_256X256_scale_minus1600_900'
dataroot = '/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256'
#for epoch_num in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
for epoch_num in [30,35,40,45,50]:
#for epoch_num in [10,20,30,40,50,60,70,80,90,100]:
#for epoch_num in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]:
    print('Removing old images from validation folder...')
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/*.npy'):
        os.remove(_f)

    print(f"------------------------- {epoch_num} MODEL CHECK ----------------------------")

    #subprocess.run(['cp', f'/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/{name}/{epoch_num}_net_G.pth', f'/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix/checkpoints/{name}/latest_net_G.pth'])

    #subprocess.run(['python3', 'test.py', '--dataroot', '/cosma5/data/durham/dc-gond1/official_pix2pix_data_F4n1_GR', '--name', f'{name}', '--model', 'bicycle_gan', '--direction', 'AtoB', '--batch_size', '1', '--input_nc', '1', '--output_nc', '1', '--num_threads=2', '--norm', 'instance', '--load_size', '512', '--crop_size', '512', '--no_flip', '--eval', '--phase', 'val', '--num_test', '120', '--results_dir', 'val_results', '--netG', 'unet_512', '--ndf', '64', '--ngf', '128', '--dataset_mode', 'aligned', '--epoch', f'{epoch_num}', '--gpu_ids', '-1', '--netE', 'conv_512', '--preprocess', 'none'])    

    subprocess.run(['python3', 'test.py', '--dataroot', f'{dataroot}', '--name', f'{name}', '--model', 'bicycle_gan', '--direction', 'AtoB', '--batch_size', '1', '--input_nc', '1', '--output_nc', '1', '--num_threads=2', '--norm', 'instance', '--load_size', '256', '--crop_size', '256', '--no_flip', '--eval', '--phase', 'val', '--num_test', '200', '--results_dir', 'val_results', '--netG', 'unet_256', '--ndf', '128', '--ngf', '128', '--dataset_mode', 'aligned', '--epoch', f'{epoch_num}', '--netE', 'resnet_256', '--preprocess', 'none', '--nef', '128', '--nz', '128', '--where_add', 'all'])

    print('Removing random_sample images from validation folder...')
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/Run*_random_sample*.npy'):
        os.remove(_f)

    result_imgs = sorted(glob.glob(f'/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/*.npy'))
    assert len(result_imgs) != 0
    for i in range(0, len(result_imgs), 3):
        assert "_encoded" in result_imgs[i]
        assert "_ground truth" in result_imgs[i+1]
        assert "_input" in result_imgs[i+2]
        fr = np.load(result_imgs[i+1]).squeeze()
        gr = np.load(result_imgs[i+2]).squeeze()
        test_curr = np.load(
                os.path.join(f'{dataroot}/val', f'{result_imgs[i+1].split("/")[-1].split("_ground truth")[0]}.npy')
        )
        orig_gr_test = test_curr[:, :256]
        orig_fr_test = test_curr[:, 256:]

        #gr = np.exp(gr*10)
        #fr = np.exp(fr*10)*gr

        #if not np.allclose(gr, orig_gr_test, atol=1e-1):
        #    print("Warning: gr and orig_gr_test do not match!")
        #if not np.allclose(fr, orig_fr_test, atol=1e-1):
        #    print("Warning: fr and orig_fr_test do not match!")

    """
    for m in grouper(12, result_imgs):
        encoded = hvstack(*[np.load(m[v]) for v in range(0, 12, 3)])
        ground_truth = hvstack(*[np.load(m[v]) for v in range(1, 12, 3)])
        inp = hvstack(*[np.load(m[v]) for v in range(2, 12, 3)])
        
        s = [m[v] for v in range(0, 12, 3)][0]  # Select the first name, which will be of the format: Run?_?_1.npy
        b = s.split('/')[-1].split('_')
        b.pop(2)  # Remove the subdivision mark
        encoded_filename = os.path.join('/'.join(s.split('/')[:-1]), '_'.join(b))
        np.save(encoded_filename, encoded)

        s = [m[v] for v in range(1, 12, 3)][0]
        b = s.split('/')[-1].split('_')
        b.pop(2)
        ground_truth_filename = os.path.join('/'.join(s.split('/')[:-1]), '_'.join(b))
        np.save(ground_truth_filename, ground_truth)

        s = [m[v] for v in range(2, 12, 3)][0]
        b = s.split('/')[-1].split('_')
        b.pop(2)
        ip_filename = os.path.join('/'.join(s.split('/')[:-1]), '_'.join(b))
        np.save(ip_filename, inp)
    

    print('Removing the 256X256 subdivision images from validation folder...')
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/Run*_*_1_*.npy'):
        os.remove(_f)
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/Run*_*_2_*.npy'):
        os.remove(_f)
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/Run*_*_3_*.npy'):
        os.remove(_f)
    for _f in glob.glob('/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/Run*_*_4_*.npy'):
        os.remove(_f)
    """

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
    results_list = sorted(glob.glob(f'/cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN/val_results/val/images/*.npy'))
    assert len(results_list) != 0
    val_gen = np.empty((int(len(results_list)/3), 256, 256))
    val_ip = np.empty((int(len(results_list)/3), 256, 256))
    val_gt = np.empty((int(len(results_list)/3), 256, 256))
    counter = 0
    for i in range(0, len(results_list), 3):
        assert "_encoded" in results_list[i]
        assert "_ground truth" in results_list[i+1]
        assert "_input" in results_list[i+2]

        gen = np.load(results_list[i]).squeeze()
        fr = np.load(results_list[i+1]).squeeze()
        gr = np.load(results_list[i+2]).squeeze()

        #gr = np.exp(gr*10)
        #gen = np.exp(gen*10)*gr
        #fr = np.exp(fr*10)*gr

        #print(gr)
        #print(fr)
        #print(gen)

        val_gen[counter] = gen
        val_ip[counter] = gr
        val_gt[counter] = fr

        counter += 1

    assert counter == int(len(results_list)/3)

    chisquare_ip_gt_median, chisquare_gen_gt_median, chisquare_ip_gt, chisquare_gen_gt, chisquare_ip_gt_PDF, chisquare_gen_gt_PDF = driver(val_gen, val_ip, val_gt, name=name, val_or_test=0, epoch_num=epoch_num,field_name='F4_veldiv', save=False, vel_field=True)

print('>> Results')
print('Median chisq pspec')
print(chisquare_gen_gt_median)
print('Mean chisq pspec')
print(chisquare_gen_gt)
print('PDF chisq distance')
print(chisquare_gen_gt_PDF)
