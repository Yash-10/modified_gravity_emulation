"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

import torch
import functools
import numpy as np
import torch.nn as nn
from torch.nn import init
import functools

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    
    print(dir(model))
    print(model.model_names)
    print(model.netD)

    model = model.netD

    from pytorch_grad_cam import DeepFeatureFactorization
    from pytorch_grad_cam.utils.image import show_factorization_on_image

    real_A = torch.from_numpy(np.load('test_results/pix2pix_F4n1_GR_DEN_a1_RESIDUAL_BUT_SCALED_LOGARITHM_TRANSFORM_NORMALIZED_BY_10/test_latest/images/Run62_300_real_A.npy').squeeze())
    real_B = torch.from_numpy(np.load('test_results/pix2pix_F4n1_GR_DEN_a1_RESIDUAL_BUT_SCALED_LOGARITHM_TRANSFORM_NORMALIZED_BY_10/test_latest/images/Run62_300_real_B.npy').squeeze())
    fake_B = torch.from_numpy(np.load('test_results/pix2pix_F4n1_GR_DEN_a1_RESIDUAL_BUT_SCALED_LOGARITHM_TRANSFORM_NORMALIZED_BY_10/test_latest/images/Run62_300_fake_B.npy').squeeze())

    real_A = real_A.unsqueeze(0).unsqueeze(0)
    real_B = real_B.unsqueeze(0).unsqueeze(0)
    fake_B = fake_B.unsqueeze(0).unsqueeze(0)

    AB = torch.cat((real_A, real_B), 1)
    import matplotlib.pyplot as plt
    output = model(AB).detach().numpy().squeeze()
    np.save('m_realA_realB.npy', output)
    plt.imshow(output); plt.colorbar(); plt.savefig('m_realA_realB.png')
    import sys
    sys.exit()
    input_tensor = AB

    #rgb_input_tensor = np.repeat(AB.squeeze()[:, :, np.newaxis], 3, axis=2)
    rgb_input_tensor = AB.squeeze().permute(1,2,0)
    print(rgb_input_tensor.shape, AB.shape)
    # rgb_input_tensor = rgb_input_tensor / rgb_input_tensor.max()

    def zero_one(img):
          return (img - img.min()) / (img.max() - img.min())

    rgb_input_tensor = (zero_one(rgb_input_tensor) * 255).type(torch.int)

    n_components = 4
    dff = DeepFeatureFactorization(model=model, target_layer=model.model.fc_added)#, computation_on_concepts=model.fc)
    concepts, batch_explanations = dff(input_tensor, n_components)
    visualization = show_factorization_on_image(rgb_input_tensor, batch_explanations[0], image_weight=0.3)
    result = np.hstack((rgb_input_tensor, visualization))

    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.savefig('res.png')
