
import numpy as np ; import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
import smoothing_library as SL


def pixel_hist(image, bins=np.logspace(start=-2, stop=2, num=99), vel_field=False):  #The choice of bins default choice: [0.01, 100] is taken from https://arxiv.org/pdf/2109.02636.pdf
    # First smooth the field.
    field = image.astype(np.float32)
    BoxSize = 128.0 #Mpc/h
    grid    = field.shape[0]
    Filter  = 'Top-Hat'
    threads = 1
    #kmin    = 0  #h/Mpc
    #kmax    = 10 #h/Mpc

    R = 10  # arbitrarily chosen by me.

    # compute the filter in Fourier space
    W_k = SL.FT_filter_2D(BoxSize, R, grid, Filter, threads)
    # smooth the field
    field_smoothed = SL.field_smoothing_2D(field, W_k, threads)

    if not vel_field:
        field_smoothed = field_smoothed / field_smoothed.mean()

    #bins = np.logspace(start=-2, stop=2, num=99)  # Bins are set according to https://browse.arxiv.org/pdf/2109.02636.pdf
    
    counts, bin_edges = np.histogram(field_smoothed, bins=bins)
    bincenters = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bincenters, counts

def pixel_pdf(images, bins=np.logspace(start=-2, stop=2, num=99), vel_field=False):
    #images is of shape: num_examples x height x width
    #num_workers = mp.cpu_count() - 1
    pixel_hist_y, pixel_hist_x = [], []

    if vel_field:
        bins = np.linspace(-900, 400, 99)

    for x in images:
        bincenters, counts = pixel_hist(x, bins=bins, vel_field=vel_field)
        pixel_hist_x.append(bincenters)
        pixel_hist_y.append(counts)

    pixel_hist_x = np.vstack(pixel_hist_x)
    pixel_hist_y = np.vstack(pixel_hist_y)

    #num_workers = 2
    #with mp.Pool(processes=num_workers) as pool:
    #    results = np.array([pool.apply(pixel_hist, (x,)) for x in images])

    #pixel_hist_y = np.vstack([y[1] for y in results])
    #pixel_hist_x = np.vstack([y[0] for y in results])
    #pixel_hist_x = np.exp(pixel_hist_x) if log else pixel_hist_x

    #x = results[0][0]

    x = np.mean(pixel_hist_x, axis=0)
    y = np.mean(pixel_hist_y, axis=0)
    xstd = np.std(pixel_hist_x, axis=0)
    ystd = np.std(pixel_hist_y, axis=0)
    ymedian = np.median(pixel_hist_y, axis=0)
    # ymad = iqr(pixel_hist_y, axis=0)
    xmedian = np.median(pixel_hist_x, axis=0)
    # xmad = iqr(pixel_hist_x, axis=0)
    return x, y, xstd, ystd, ymedian, xmedian


x4 = []
for img in sorted(glob.glob('official_pix2pix_data_velDiv_F4n1_GR_256X256/train/*.npy'))[:2000]:
    arr = np.load(img)[:, 256:]

    # arr[arr>scale]=scale
    # arr[arr<-scale]=-scale

    x4.append(arr)

pdf4 = pixel_pdf(x4, vel_field=True)

x4_test = []
for img in sorted(glob.glob('official_pix2pix_data_velDiv_F4n1_GR_256X256/test/*.npy'))[:2000]:
    arr = np.load(img)[:, 256:]

    # arr[arr>scale]=scale
    # arr[arr<-scale]=-scale

    x4_test.append(arr)

pdf4_test = pixel_pdf(x4_test, vel_field=True)

x6_test = []
for img in sorted(glob.glob('official_pix2pix_data_velDiv_F6n1_GR_256X256/test/*.npy'))[:2000]:
    arr = np.load(img)[:, 256:]

    # arr[arr>scale]=scale
    # arr[arr<-scale]=-scale

    x6_test.append(arr)

pdf6_test = pixel_pdf(x6_test, vel_field=True)

x5_test = []
for img in sorted(glob.glob('official_pix2pix_data_velDiv_F5n1_GR_256X256/test/*.npy'))[:2000]:
    arr = np.load(img)[:, 256:]

    # arr[arr>scale]=scale
    # arr[arr<-scale]=-scale

    x5_test.append(arr)

pdf5_test = pixel_pdf(x5_test, vel_field=True)


plt.plot(pdf4[0], pdf4[1], label='F4-train')
plt.plot(pdf4_test[0], pdf4_test[1], label='F4-test')
plt.plot(pdf5_test[0], pdf5_test[1], label='F5-test')
plt.plot(pdf6_test[0], pdf6_test[1], label='F6-test')
plt.legend()
plt.yscale('log')
plt.savefig('velfield_ALL_PLOT.png')

