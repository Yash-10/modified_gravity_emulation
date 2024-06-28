import numpy as np
import glob

from astropy.constants import G
G = G.value
# G = 1
a = 1.0  # Scale factor

# ******** SEE https://github.com/grkooij/Cosmological-Particle-Mesh-Simulation/blob/master/src/configure_me.py ********
N_PARTS          = 256 # Number of particles in one dimension
BOX_SIZE         = 128 # Box size in Mpc/h in one dimension
N_CELLS          = 256 # Number of cells in one dimension
N_CPU            = 1 # Number of cpu's used

def get_potential(density):
    """This is for the Newtonian gravity.

    See GLAMdoc.pdf. The approach used in this function is the simples way
    to related matter densities to the gravitational potential. But it may
    not be the most accurate.
    """
    density_mean = density.mean()
    density_contrast = density / density.mean() - 1

    density_k = np.fft.fftn(density_contrast)

    scale = 2*np.pi*N_PARTS/BOX_SIZE
    lxaxis = scale*np.fft.fftfreq(N_PARTS)
    lyaxis = scale*np.fft.fftfreq(N_PARTS)

    #2D Fourier axes
    ly, lx = np.meshgrid(lyaxis, lxaxis, indexing='ij')

    #-k squared operator where k = sqrt(lx**2 + ly**2)
    del_sq = -(lx**2 + ly**2)

    #Calculating potential and correcting for scale with mass resolution
    scaled_density_k = density_k * (4 * np.pi * G * a**2 * density_mean)
    # As mentioned in Page 16 of https://astro.uchicago.edu/~andrey/Talks/PM/pm.pdf,
    # singularity during division is handled by manually setting the potential to zero when |k| = 0.
    # That's why we input the `out` argument.
    potential_k = np.divide(scaled_density_k, del_sq, where=del_sq!=0, out=np.zeros_like(scaled_density_k))

    potential_real = np.fft.ifftn(potential_k).real
    return potential_real

def get_screening_map(density_gr, density_fr):
    potential_gr = get_potential(density_gr)
    potential_fr = get_potential(density_fr)
    force_gr = np.sqrt(np.gradient(potential_gr, axis=0) ** 2 + np.gradient(potential_gr, axis=1) ** 2)
    force_fr = np.sqrt(np.gradient(potential_fr, axis=0) ** 2 + np.gradient(potential_fr, axis=1) ** 2)

    return force_fr / force_gr

