import numpy as np
import gzip
import os

counter = 0
fr_prefix = 'F5'
#for prefix in ['Run1', 'Run2', 'Run10']:
#for prefix in ['Run30']:
for prefix in ['Run62']:
    gr_gzip = f'example_GR_L128Np256Ng512_{prefix}_0157_velDiv.npy.gz'
    fr_gzip = f'example_{fr_prefix}n1_L128Np256Ng512_{prefix}_0157_velDiv.npy.gz'

    Ng = 512
    base_path = f'/cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_{fr_prefix}n1_GR/'

    gr_file = gzip.GzipFile(gr_gzip, 'r')
    fr_file = gzip.GzipFile(fr_gzip, 'r')

    gr = np.load(gr_file); fr = np.load(fr_file)

    counter = 0
    for i in range(0, Ng):
        gr_ = gr[:, :, i]; fr_ = fr[:, :, i]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, 'test', f'{prefix}_{counter}.npy'), combined)

        counter += 1

        gr_ = gr[:, i, :]; fr_ = fr[:, i, :]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, 'test', f'{prefix}_{counter}.npy'), combined)

        counter += 1

        gr_ = gr[i, :, :]; fr_ = fr[i, :, :]
        combined = np.hstack((gr_, fr_))
        np.save(os.path.join(base_path, 'test', f'{prefix}_{counter}.npy'), combined)

        counter += 1

