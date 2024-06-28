
# Template code to save data for the machine learning application.

import numpy as np
import gzip
import os

counter = 0
fr_prefix = 'F6'

for prefix in ['Run1', 'Run2', 'Run30', 'Run62', 'Run100']:
    gr_gzip = f'example_GR_L128Np256Ng512_{prefix}_0157_velDiv.npy.gz'
    fr_gzip = f'example_{fr_prefix}n1_L128Np256Ng512_{prefix}_0157_velDiv.npy.gz'

    Ng = 512
    base_path = f'/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_{fr_prefix}n1_GR/'

    gr_file = gzip.GzipFile(gr_gzip, 'r')
    fr_file = gzip.GzipFile(fr_gzip, 'r')

    gr = np.load(gr_file); fr = np.load(fr_file)

    if prefix == 'Run1' or prefix == 'Run2' or prefix == 'Run30':
        dir_ = 'train'
    elif prefix == 'Run62':
        dir_ = 'val'
    elif prefix == 'Run100':
        dir_ = 'test'

    counter = 0
    for i in range(0, Ng):
        gr_ = gr[:, :, i]; fr_ = fr[:, :, i]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, dir_, f'{prefix}_{counter}.npy'), combined)

        counter += 1

        gr_ = gr[:, i, :]; fr_ = fr[:, i, :]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, dir_, f'{prefix}_{counter}.npy'), combined)

        counter += 1

        gr_ = gr[i, :, :]; fr_ = fr[i, :, :]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, dir_, f'{prefix}_{counter}.npy'), combined)

        counter += 1


# Below code is to save 256x256 2D slices of the 512x512 2D density fields, which is used in this work.

import glob
for d in ['train', 'val', 'test']:
    for img in glob.glob(f'official_pix2pix_data_velDiv_{fr_prefix}n1_GR/{d}/*.npy'):
        #print(img)
        name = img.split('/')[-1].split('.')[0]
        #print(name)
        arr = np.load(img)
        gr = arr[:, :512]
        fr = arr[:, 512:]
        gr_1 = gr[:256, :256]
        gr_2 = gr[:256, 256:]
        gr_3 = gr[256:, :256]
        gr_4 = gr[256:, 256:]
        fr_1 = fr[:256, :256]
        fr_2 = fr[:256, 256:]
        fr_3 = fr[256:, :256]
        fr_4 = fr[256:, 256:]
        np.save(f'official_pix2pix_data_velDiv_{fr_prefix}n1_GR_256X256/{d}/{name}_1.npy', np.hstack((gr_1, fr_1)))
        np.save(f'official_pix2pix_data_velDiv_{fr_prefix}n1_GR_256X256/{d}/{name}_2.npy', np.hstack((gr_2, fr_2)))
        np.save(f'official_pix2pix_data_velDiv_{fr_prefix}n1_GR_256X256/{d}/{name}_3.npy', np.hstack((gr_3, fr_3)))
        np.save(f'official_pix2pix_data_velDiv_{fr_prefix}n1_GR_256X256/{d}/{name}_4.npy', np.hstack((gr_4, fr_4)))
