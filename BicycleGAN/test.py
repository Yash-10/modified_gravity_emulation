import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import numpy as np
import random

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


#random.seed(42)
#nums = np.append(random.sample(range(1536), 500), 300)  # for test
#nums = random.sample(range(1536), 200)  # for val
#print(nums)

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)

    #img_path = model.get_image_paths()
    #if not any('Run62_'+str(num)+'_' in img_path[0] for num in nums):
    #    continue
    #if not 'Run62_334_' in img_path[0]:
    #    continue

    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

    #img_path = 'input_%3.3d' % i
    #img_path = model.get_image_paths()
    #np.random.seed(42)
    #nums = np.append(np.random.choice(range(1536), 500), 300)
    #print(nums)
    #if any('Run62_'+str(num)+'_' in img_path[0] for num in nums):
    #    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    img_path = model.get_image_paths()
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    #if any('Run62_'+str(num)+'_' in img_path[0] for num in np.append(np.random.choice(range(1536), 500), 300)):
    #    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

webpage.save()

