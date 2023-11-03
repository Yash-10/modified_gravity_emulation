#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH -J ml_mg_train
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 48:00:00
#SBATCH --exclude=mad06,mad05
#SBATCH --mail-type=END
#SBATCH --mail-user=f20190481@goa.bits-pilani.ac.in

#load the modules used to build your program.
#module purge
#module load cuda/11.2
module unload python/3.6.5
module load python/3.8.7-C8
nvcc --version
nvidia-smi

cd /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt

# Run the program
pwd
cd /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/pytorch-CycleGAN-and-pix2pix
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_F6n1_GR/train | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_F6n1_GR/val | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_F6n1_GR/test | wc -l

python3 train.py --dataset_mode aligned --dataroot /cosma5/data/durham/dc-gond1/official_pix2pix_data_F6n1_GR --name pix2pix_F6n1_GR_DEN_a1_RESIDUAL_BUT_SCALED_LOGARITHM_TRANSFORM_NORMALIZED_BY_10_Focal_Freq_Loss_and_L1_Patch4_and_NEW_MODIFIED_MODIFIED --model pix2pix --direction AtoB \
                      --batch_size 1 --input_nc 1 --output_nc 1 --lambda_L1 100 --netG unet_512 --load_size 512 --crop_size 512 --num_threads=2 --norm instance \
                                        --display_id 0 --n_epochs 0 --n_epochs_decay 200 --lr 2e-4 --save_latest_freq 800000 --save_epoch_freq 10 --netD n_layers --n_layers_D 4 --preprocess none\
                                                          --ndf 64 --ngf 128 --display_freq 10000000 --lr_policy step --lr_decay_iters 50 --phase train

