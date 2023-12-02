import glob
import numpy as np
import matplotlib.pyplot as plt

bins = np.logspace(start=-2, stop=4, num=99)

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, :512])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_GR.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F4n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F4.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F5n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F5.png', bbox_inches='tight', dpi=200)
plt.close()





arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/train/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='train', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')

arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/val/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='val', alpha=0.9, density=True, histtype='step')

arrs = []
for img in glob.glob('official_pix2pix_data_F6n1_GR/test/*.npy'):
    arr = np.load(img)
    arrs.append(arr[:, 512:])

arrs = np.array(arrs)
plt.hist(arrs.ravel(), bins=bins, label='test', alpha=0.9, density=True, histtype='step')
plt.yscale('log'); plt.xscale('log')
plt.legend()
plt.savefig('densities_F6.png', bbox_inches='tight', dpi=200)
plt.close()

