import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
from argparse import Namespace

opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle


opt2 = Namespace(**vars(opt))
# CHANGE: Write the name of the dataroot on which you trained the model OR the dataroot on which the model performed well/is expected to perform well.
opt2.dataroot = '/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256'
scale_factor = 25.0

# create dataset
dataset = create_dataset(opt)
dataset2 = create_dataset(opt2)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))


zrA_list, zrB_list, zrB2_list, zfB_list, zfB2_list, A_paths_list = [], [], [], [], [], []


# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# test stage
for i, (data, data2) in enumerate(zip(
        islice(dataset, opt.num_test),
        islice(dataset2, opt2.num_test)
        )
    ):
    model.set_input(input=data, input2=data2, latent_interpolate=True)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B, fake_B2, zrA, zrB, zfB, zrB2, zfB2 = model.test(z_samples[[nn]], encode=encode, latent_interpolate=True, scale_factor=scale_factor)
        if nn == 0:
            images = [real_A, real_B, fake_B, fake_B2]
            names = ['input', 'ground truth', 'encoded', f'encoded_latent_interp_sf{scale_factor}']
            # zrA: GR-sim, zrB: F6-sim, zrB2: F4-sim, zfB: F6_gen, zfB2: F6_gen_latent_interp
            zrA_list.append(zrA.cpu().numpy())
            zrB_list.append(zrB.cpu().numpy())
            zrB2_list.append(zrB2.cpu().numpy())
            zfB_list.append(zfB.cpu().numpy())
            zfB2_list.append(zfB2.cpu().numpy())
            A_paths_list.append(data['A_paths'])
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)
        break  # Break since we are only interested in encode=True and not the random samples.

    #img_path = 'input_%3.3d' % i
    img_path = model.get_image_paths()
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()


import numpy as np
np.save('zrA_list.npy', np.array(zrA_list))
np.save('zrB_list.npy', np.array(zrB_list))
np.save('zfB_list.npy', np.array(zfB_list))
np.save('zrB2_list.npy', np.array(zrB2_list))
np.save('zfB2_list.npy', np.array(zfB2_list))
np.save('A_paths_list.npy', np.array(A_paths_list))

