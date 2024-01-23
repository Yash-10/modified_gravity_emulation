import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html


# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

zrA_list, zrB_list, zfB_list, A_paths_list = [], [], [], []
# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        #real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn != 0:
            continue
        real_A, fake_B, real_B, zrA, zrB, zfB = model.test(z_samples[[nn]], encode=encode)
        zrA_list.append(zrA.numpy())
        zrB_list.append(zrB.numpy())
        #zfB_list.append(zfB.numpy())
        A_paths_list.append(data['A_paths'])
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

    #img_path = 'input_%3.3d' % i
    #save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()

import numpy as np
np.save('zrA_list.npy', np.array(zrA_list))
np.save('zrB_list.npy', np.array(zrB_list))
#np.save('zfB_list.npy', np.array(zfB_list))
np.save('A_paths_list.npy', np.array(A_paths_list))
