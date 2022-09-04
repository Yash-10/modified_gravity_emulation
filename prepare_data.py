import os
import shutil
import glob
import numpy as np
import gzip


def unison_shuffled_copies(a, b):
    """Taken from https://stackoverflow.com/a/4602224"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_slice(arr_3d, i):
    _l = arr_3d[i].split('/')[2]
    a = gzip.GzipFile(_l.split('gz')[0]+'gz', 'r')
    gr_3d = np.load(a)
    print(_l)
    if _l[6] == 'z':
        arr_slice = gr_3d[:, :, int(_l.split('_')[7])]
    elif _l[1] == 'y':
        arr_slice = gr_3d[:, int(_l.split('_')[7]), :]
    elif _l[1] == 'x':
        arr_slice = gr_3d[int(_l.split('_')[7]), :, :]

    return arr_slice

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    BASE_PATH = '/content'

    if os.path.exists(os.path.join(BASE_PATH, 'gr')):
        shutil.rmtree(os.path.join(BASE_PATH, 'gr'))
    if os.path.exists(os.path.join(BASE_PATH, 'f4')):
        shutil.rmtree(os.path.join(BASE_PATH, 'f4'))
 
    os.mkdir(os.path.join(BASE_PATH, 'gr'))
    os.mkdir(os.path.join(BASE_PATH, 'f4'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'train'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'val'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'test'))
    os.mkdir(os.path.join(BASE_PATH, 'f4', 'train'))
    os.mkdir(os.path.join(BASE_PATH, 'f4', 'val'))
    os.mkdir(os.path.join(BASE_PATH, 'f4', 'test'))

    Ng = 512  # Grid size.
    assert len(glob.glob(f'{BASE_PATH}/example_GR_*_*den.npy.gz')) == len(glob.glob(f'{BASE_PATH}/example_F4n1_*_*den.npy.gz'))
    total_slices = Ng * 3 * len(glob.glob(f'{BASE_PATH}/example_GR_*_*_0157_den.npy.gz'))  # These many slices for either GR or F4n1.
    gr_all = ['dummy'] * total_slices
    fr_all = ['dummy'] * total_slices

    counter = 0
    for run in ['Run1', 'Run2', 'Run62']:  # Assumes three runs with the given names are only present.
        gr_npy_gz_file = f'{BASE_PATH}/example_GR_L128Np256Ng512_{run}_0157_den.npy.gz'
        g = gzip.GzipFile(gr_npy_gz_file, 'r')
        fr_npy_gz_file = f'{BASE_PATH}/example_F4n1_L128Np256Ng512_{run}_0157_den.npy.gz'
        f = gzip.GzipFile(fr_npy_gz_file, 'r')

        for i in range(0, Ng):
            gr_all[counter] = g.name + f'_z_{i}'
            fr_all[counter] = f.name + f'_z_{i}'

            counter += 1

            gr_all[counter] = g.name + f'_y_{i}'
            fr_all[counter] = f.name + f'_y_{i}'

            counter += 1

            gr_all[counter] = g.name + f'_x_{i}'
            fr_all[counter] = f.name + f'_x_{i}'

            counter += 1

    assert len(gr_all) == total_slices
    assert len(fr_all) == total_slices
    assert 'dummy' not in gr_all
    assert 'dummy' not in fr_all

    # print(gr_all)

    gr_all, fr_all = unison_shuffled_copies(np.array(gr_all), np.array(fr_all))
    rows = len(gr_all)

    # Perform 80-10-10% split for train-val-test.
    train_indices = range(int(np.round(0.8 * rows)))
    val_index_start = train_indices[-1] + 1
    num_val = int(np.round(0.1 * rows))
    val_index_end = val_index_start+num_val
    val_indices = range(rows)[val_index_start:val_index_end]
    assert len(val_indices) == num_val
    test_index_start = val_indices[-1] + 1
    num_test = int(np.round(0.1 * rows))
    test_index_end = test_index_start+num_test
    test_indices = range(rows)[test_index_start:test_index_end]

    assert set(train_indices).isdisjoint(set(val_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(val_indices).isdisjoint(set(test_indices))

    # Save train
    for i, index in enumerate(train_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'train', f'{i}.npy'), get_slice(gr_all, i))
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'train', f'{i}.npy'), get_slice(fr_all, i))

    # Save val
    for i, index in enumerate(val_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'val', f'{i}.npy'), get_slice(gr_all, i))
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'val', f'{i}.npy'), get_slice(fr_all, i))

    # Save test
    for i, index in enumerate(test_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'test', f'{i}.npy'), get_slice(gr_all, i))
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'test', f'{i}.npy'), get_slice(fr_all, i))
