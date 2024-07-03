# Emulation of f(R) modified gravity from $\Lambda$ CDM using conditional GANs

This repository contains code for developing a field-level, neural-network-based emulator that generates density and velocity divergence fields under the f(R) gravity MG model from the corresponding $\Lambda$ CDM simulated fields.

The code here is a modification to the [BicycleGAN](https://junyanz.github.io/BicycleGAN/) approach for multimodal image-to-image translation; the official BicycleGAN codebase is [here](https://github.com/junyanz/BicycleGAN) and the paper on BicycleGAN is [here](https://arxiv.org/abs/1711.11586). Our modifications to the BicycleGAN code are located [here](https://github.com/Yash-10/modified_gravity_emulation/tree/main/BicycleGAN).

## Motivation

Testing deviations from the $\Lambda$ CDM cosmological model is a key science target of astronomical surveys. However, the simulations of these alternatives are immensely slow. We use the f(R) gravity model in this study as an example, which is a widely studied modified gravity (MG) theory.

![Simulation execution time](https://github.com/Yash-10/modified_gravity_emulation/blob/main/imgs/sim_exec_time.png)

To alleviate the computational bottleneck imposed by f(R) gravity simulations, we use a GAN conditioned on the outputs of $\Lambda$ CDM simulations that emulate f(R) simulation outputs. We focus on the matter density and velocity divergence fields.

## Scientific details

Coming soon...

## Installation

The installation instructions are the same as mentioned in the [original BicycleGAN](https://github.com/junyanz/BicycleGAN?tab=readme-ov-file#installation) repository.

## Usage

### Preparing the data
- Our approach requires aligned pairs of images, one from the $\Lambda$ CDM simulation and the other from the f(R) simulation for training the model. The simulation data in this study is obtained from $N$-body simulations of the MG-GLAM code.
- We have then used [DTFE](https://github.com/MariusCautun/DTFE) to interpolate the particle positions onto a uniform grid. The resulting density fields can be stored in various formats; consult the [DTFE documentation](https://github.com/MariusCautun/DTFE/blob/master/documentation/DTFE_user_guide.pdf) for details; here we save them as `npy.gz` files, which are read in the script `scripts/prepare_data.py`. An example command used is:

```bash
./DTFE example_F4n1_L128Np256Ng512_Run1_0157.hdf5 test --grid 512 -i 105 --output 101 --field density_a velocity_a divergence_a --periodic
```

- The script `scripts/prepare_data.py` provides a template code to arrange the simulation data in a manner that can be used with the BicycleGAN code. It reads the DTFE densities (which we have processed into `npy.gz` files) and combines the GR and the corresponding f(R) 2D fields (in this study, we use 512x512 grid size when using DTFE) into a single array and saves them. For example, with GR and F6 velocity divergence data, this script will store numpy binary files (`.npy`) in `official_pix2pix_data_velDiv_F6n1_GR/train`, `official_pix2pix_data_velDiv_F6n1_GR/val`, and `official_pix2pix_data_velDiv_F6n1_GR/test` directories. There is code to additionally save 256x256 cutouts from the 512x512 fields, which is used in this work.

**Note**: `prepare_data.py` hardcodes the code assuming GR and F6 velocity divergence, `velDiv`, is used, but the code can be modified for any other case, such as GR and F4 density. More details about the simulation can be found in our paper (see the [Scientific details section](https://github.com/Yash-10/modified_gravity_emulation?tab=readme-ov-file#scientific-details)).

### Training, validation, and testing
- The script `scripts/train.sh` can be used for training (which itself runs `BicycleGAN/train.py`), and contains the hyperparameters used. During training, paired simulated fields from $\Lambda$ CDM and f(R) gravity are used. 
- `scripts/testing.sh` can be used for testing the model - uncomment the line that needs to be run for testing (i.e., if you want to run the script `testing_for_den_F4den_test_Bicycle.py`, uncomment the corresponding line in `scripts/testing.sh`). The description of some of the testing python scripts is as follows (for the below description, we assume the model is trained to learn the $\Lambda$ CDM to F4 mapping; F4 means f(R) with $|f_{R0}| = 10^{-4}$, see the paper for detailed description):
    - `testing_for_den_F4den_test_Bicycle.py`: for testing the model on the $\Lambda$ CDM-F4 case.
    - `testing_for_den_F4den_test_Bicycle_latent_interp.py`: for performing latent extrapolation for $\Lambda$ CDM-F5 / $\Lambda$ CDM-F6 cases (change `dataroot` accordingly). Note that this script does not in itself perform latent extrapolation, it runs `BicycleGAN/test_latent_interpolate.py` instead of `BicycleGAN/test.py`. Latent extrapolation is added as an option in `BicycleGAN/models/bicycle_gan_model.py` called `latent_interpolate` (the name `latent_interpolate` should actually have been `latent_extrapolate` since we perform latent extrapolation and latent interpolation, so this name is a slight mismatch).
    - `testing_for_den_F4den_validation_Bicycle.py`: for validating the model on the $\Lambda$ CDM-F4 case. This will run all saved checkpoints that were stored during training on the validation data.
    - `testing_for_den_F4den_validation_Bicycle_latent_interp.py`: for validating the model when using latent extrapolation for the $\Lambda$ CDM-F5 / $\Lambda$ CDM-F6 cases.

and similarly for the velocity divergence.
- At test time, given a simulated $\Lambda$ CDM field, the model will generate the corresponding f(R) field. In BicycleGAN, the generator network takes two inputs: the input field (from $\Lambda$ CDM, in this case) and a latent vector. In our study, we had the simulated f(R) fields (ground-truth) available to evaluate the model's prediction, so the latent vector used during test time was obtained from encoding the ground-truth through the encoder network component of BicycleGAN, which is also trained during training. However, if the ground truth is unavailable, the latent vector can be a randomly sampled vector from some known distribution, such as Gaussian. The BicycleGAN code contains options to handle this. For example, see the [`get_z_random` method used in this line](https://github.com/Yash-10/modified_gravity_emulation/blob/4e96cfde58ad9b0275d80bf6768daac4b8e43328/BicycleGAN/models/bicycle_gan_model.py#L112) as a starting point.
- The evaluation metrics used to select the best model based on the validation set and for evaluating the selected model on the test set can be found in `scripts/evaluation_metrics.py`. To select the best model, we select the model that generally performs well on the metrics used in this study (2D power spectrum, histogram of values, cumulants of the distribution). This script contains the `driver` function, which is used in all the validation/testing scripts described above.

### Interpretation
- `BicycleGAN/spat_attn_scalar_field.py` contains code to store some intermediate outputs of the discriminator network of the BicycleGAN model. These outputs are then used in `BicycleGAN/interpretation_with_scalar_field.ipynb` to visualize the attention weights learned by the discriminator network.
- Note that `BicycleGAN/interpretation_with_scalar_field.ipynb` also uses `output_F6_Run62.dat`, which is the full (512x512x512) 3D scalar field array for a particular run (Run62) of the F6 simulations of MG-GLAM. The scalar field is obtained from the MG-GLAM simulation code. Due to its large disk usage (~512 MB), it is not stored here.
- The notebook visually compares the attention weights and the scalar field for possible correlations (see the paper for detailed discussion).

### Execution time
- The script used to calculate the inference time of the emulator is `BicycleGAN/time_bicyclegan.py`. The script loads the trained generator model and calculates the time required for a forward pass in inference mode using a CPU. `BicycleGAN/timing_logs.txt` contains some logs of the execution time of the simulation and our emulator.


## Bugs or issues
If you find something not working as expected or want to discuss a feature, we would like to know about it. Please feel free to open an issue in the [issue tracker](https://github.com/Yash-10/modified_gravity_emulation/issues) or [send an email](yashgondhalekar567@gmail.com).

## License
[MIT](https://github.com/Yash-10/modified_gravity_emulation/blob/main/LICENSE)
