import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import matplotlib.pyplot as plt

import numpy as np
import torch
from timeit import default_timer as timer

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers

model.eval()

dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

times = []
for i, data in enumerate(dataset):
    #if data['A_paths'][0] != '/cosma5/data/durham/dc-gond1/official_pix2pix_data_F4n1_GR/test/Run62_300.npy':
    #    continue
    start = timer()
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    end = timer()
    print(f'Time required (s): {end - start}')
    times.append(end - start)
    #visuals = model.get_current_visuals()  # get image results
    #img_path = model.get_image_paths()     # get image paths

    """
    gr = np.exp(visuals['real_A']*10)
    gen = np.exp(visuals['fake_B']*10)*visuals['real_A']
    fr = np.exp(visuals['real_B']*10)*visuals['real_A']

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(gr.squeeze()); ax[0].set_title('GR sim')
    ax[1].imshow(fr.squeeze()); ax[1].set_title('F4 sim')
    ax[2].imshow(gen.squeeze()); ax[2].set_title('F4 cGAN')
    plt.savefig('time_calculation_visual.png', bbox_inches='tight', dpi=200)
    break
    """
print(i)
print(np.sum(times))
print(np.mean(times))
print(np.median(times))
# todo: include CPU or GPU time? Or both?
