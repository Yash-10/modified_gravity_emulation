import os
import glob
import numpy as np
import gzip


def unison_shuffled_copies(a, b):
    """Taken from https://stackoverflow.com/a/4602224"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)

    BASE_PATH = '/content'

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
    total_slices = Ng * 3 * len(glob.glob(f'{BASE_PATH}/example_GR_*_*den.npy.gz'))  # These many slices for either GR or F4n1.

    gr_all = np.empty((total_slices, Ng, Ng))
    fr_all = np.empty((total_slices, Ng, Ng))

    counter = 0
    for run in ['Run1', 'Run2', 'Run62']:  # Assumes three runs with the given names are only present.
        gr_npy_gz_file = f'{BASE_PATH}/example_GR_*{run}_*den.npy.gz'
        g = gzip.GzipFile(gr_npy_gz_file, 'r')
        gr = np.load(g)
        fr_npy_gz_file = f'{BASE_PATH}/example_GR_*{run}_*den.npy.gz'
        f = gzip.GzipFile(fr_npy_gz_file, 'r')
        fr = np.load(f)

        for i in range(0, Ng):
            gr_arr_slice = gr[:, :, i]
            f4n1_arr_slice = fr[:, :, i]
            gr_all[counter] = gr_arr_slice
            fr_all[counter] = f4n1_arr_slice

            counter += 1

            gr_arr_slice = gr[:, i, :]
            f4n1_arr_slice = fr[:, i, :]
            gr_all[counter] = gr_arr_slice
            fr_all[counter] = f4n1_arr_slice

            counter += 1

            gr_arr_slice = gr[i, :, :]
            f4n1_arr_slice = fr[i, :, :]
            gr_all[counter] = gr_arr_slice
            fr_all[counter] = f4n1_arr_slice

            counter += 1

    assert gr_all.shape == (total_slices, Ng, Ng)
    assert fr_all.shape == (total_slices, Ng, Ng)

    gr_all, fr_all = unison_shuffled_copies(gr_all, fr_all)
    rows = gr_all.shape[0]

    # Perform 80-10-10% split for train-val-test.
    train_indices = range(np.round(0.8 * rows))
    val_index_start = train_indices[-1] + 1
    num_val = np.round(0.1 * rows)
    val_indices = rows[val_index_start:val_index_start+num_val]
    assert len(val_indices) == num_val
    test_index_start = val_indices[-1] + 1
    num_test = np.round(0.1 * rows)
    test_indices = rows[test_index_start:test_index_start+num_test]

    assert set(train_indices).isdisjoin(set(val_indices))
    assert set(train_indices).isdisjoin(set(test_indices))
    assert set(val_indices).isdisjoin(set(test_indices))

    # Save train
    for i, index in enumerate(train_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'train', f'{i}.npy'), gr_all[i])
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'train', f'{i}.npy'), fr_all[i])

    # Save val
    for i, index in enumerate(val_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'val', f'{i}.npy'), gr_all[i])
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'val', f'{i}.npy'), fr_all[i])

    # Save test
    for i, index in enumerate(test_indices):
        # Save GR
        np.save(os.path.join(BASE_PATH, 'gr', 'test', f'{i}.npy'), gr_all[i])
        # Save FR
        np.save(os.path.join(BASE_PATH, 'fr', 'test', f'{i}.npy'), fr_all[i])