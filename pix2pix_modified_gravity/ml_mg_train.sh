#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH -J ml_mg_f5
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=f20190481@goa.bits-pilani.ac.in

#load the modules used to build your program.
module purge
module load cuda/11.2
module unload python/3.6.5
module load python/3.8.7-C8
nvcc --version
nvidia-smi

cd /cosma5/data/durham/dc-gond1/modified_pix2pix/pytorch-CycleGAN-and-pix2pix
pip install -r requirements.txt

# Run the program
pwd
cd /cosma5/data/durham/dc-gond1/modified_pix2pix/pytorch-CycleGAN-and-pix2pix
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data/train | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data/val | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data/test | wc -l

python3 train.py --dataset_mode aligned --dataroot /cosma5/data/durham/dc-gond1/official_pix2pix_data_F5n1_GR --name pix2pix_F5n1_GR --model pix2pix --direction AtoB \
                      --batch_size 1 --input_nc 1 --output_nc 1 --lambda_L1 200 --netG unet_512 --load_size 542 --crop_size 512 --num_threads=2 --norm instance \
                                        --display_id 0 --n_epochs 300 --lr 2e-4 --save_latest_freq 147440 --save_epoch_freq 10 --netD n_layers --n_layers_D 4 \
                                                          --ndf 64 --ngf 128 --display_freq 10000000 --lr_policy cosine
