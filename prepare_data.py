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
    if os.path.exists(os.path.join(BASE_PATH, 'fr')):
        shutil.rmtree(os.path.join(BASE_PATH, 'fr'))
 
    os.mkdir(os.path.join(BASE_PATH, 'gr'))
    os.mkdir(os.path.join(BASE_PATH, 'fr'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'train'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'val'))
    os.mkdir(os.path.join(BASE_PATH, 'gr', 'test'))
    os.mkdir(os.path.join(BASE_PATH, 'fr', 'train'))
    os.mkdir(os.path.join(BASE_PATH, 'fr', 'val'))
    os.mkdir(os.path.join(BASE_PATH, 'fr', 'test'))

    Ng = 512  # Grid size.
    assert len(glob.glob(f'{BASE_PATH}/example_GR_*_*den.npy.gz')) == len(glob.glob(f'{BASE_PATH}/example_F5n1_*_*den.npy.gz'))
    total_slices = Ng * 3 * len(glob.glob(f'{BASE_PATH}/example_GR_*_*_0157_den.npy.gz'))  # These many slices for either GR or F5n1.

    # Perform 80-10-10% split for train-val-test.
    train_indices = range(int(np.round(0.8 * total_slices)))
    val_index_start = train_indices[-1] + 1
    num_val = int(np.round(0.1 * total_slices))
    val_index_end = val_index_start+num_val
    val_indices = range(total_slices)[val_index_start:val_index_end]
    assert len(val_indices) == num_val
    test_index_start = val_indices[-1] + 1
    num_test = int(np.round(0.1 * total_slices))
    test_index_end = test_index_start+num_test
    test_indices = range(total_slices)[test_index_start:test_index_end]

    assert set(train_indices).isdisjoint(set(val_indices))
    assert set(train_indices).isdisjoint(set(test_indices))
    assert set(val_indices).isdisjoint(set(test_indices))
    assert len(train_indices) + len(val_indices) + len(test_indices) == total_slices
    
    counter = 0
    for run in ['Run1', 'Run2', 'Run62']:  # Assumes three runs with the given names are only present.
        gr_npy_gz_file = f'{BASE_PATH}/example_GR_L128Np256Ng512_{run}_0157_den.npy.gz'
        g = gzip.GzipFile(gr_npy_gz_file, 'r')
        gr = np.load(g)
        fr_npy_gz_file = f'{BASE_PATH}/example_F5n1_L128Np256Ng512_{run}_0157_den.npy.gz'
        f = gzip.GzipFile(fr_npy_gz_file, 'r')
        fr = np.load(f)

        for i in range(0, Ng):
            if counter in train_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'train', f'{run}_{counter}.npy'), gr[:, :, i])
                np.save(os.path.join(BASE_PATH, 'fr', 'train', f'{run}_{counter}.npy'), fr[:, :, i])
            elif counter in val_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'val', f'{run}_{counter}.npy'), gr[:, :, i])
                np.save(os.path.join(BASE_PATH, 'fr', 'val', f'{run}_{counter}.npy'), fr[:, :, i])
            elif counter in test_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'test', f'{run}_{counter}.npy'), gr[:, :, i])
                np.save(os.path.join(BASE_PATH, 'fr', 'test', f'{run}_{counter}.npy'), fr[:, :, i])

            counter += 1

            if counter in train_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'train', f'{run}_{counter}.npy'), gr[:, i, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'train', f'{run}_{counter}.npy'), fr[:, i, :])
            elif counter in val_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'val', f'{run}_{counter}.npy'), gr[:, i, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'val', f'{run}_{counter}.npy'), fr[:, i, :])
            elif counter in test_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'test', f'{run}_{counter}.npy'), gr[:, i, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'test', f'{run}_{counter}.npy'), fr[:, i, :])

            counter += 1

            if counter in train_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'train', f'{run}_{counter}.npy'), gr[i, :, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'train', f'{run}_{counter}.npy'), fr[i, :, :])
            elif counter in val_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'val', f'{run}_{counter}.npy'), gr[i, :, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'val', f'{run}_{counter}.npy'), fr[i, :, :])
            elif counter in test_indices:
                np.save(os.path.join(BASE_PATH, 'gr', 'test', f'{run}_{counter}.npy'), gr[i, :, :])
                np.save(os.path.join(BASE_PATH, 'fr', 'test', f'{run}_{counter}.npy'), fr[i, :, :])

            counter += 1

    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'gr', 'train')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'gr', 'train'), name))]))
    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'gr', 'val')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'gr', 'val'), name))]))
    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'gr', 'test')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'gr', 'test'), name))]))
    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'fr', 'train')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'fr', 'train'), name))]))
    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'fr', 'val')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'fr', 'val'), name))]))
    print(len([name for name in os.listdir(os.path.join(BASE_PATH, 'fr', 'test')) if os.path.isfile(os.path.join(os.path.join(BASE_PATH, 'fr', 'test'), name))]))
