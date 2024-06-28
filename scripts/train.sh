#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH -J ml_mg_train
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH --exclude=mad04,mad05
#SBATCH -A durham
#SBATCH -t 48:00:00

#load the modules used to build your program.
#module purge
#module load cuda/11.2
module unload python/3.6.5
module load python/3.8.7-C8
nvidia-smi

cd /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN
pip install -r requirements.txt

# Run the program
pwd
cd /cosma5/data/durham/dc-gond1/pix2pix_modified_gravity/BicycleGAN
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256/train | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256/val | wc -l
ls /cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256/test | wc -l

python3 train.py --dataset_mode aligned --dataroot /cosma5/data/durham/dc-gond1/official_pix2pix_data_velDiv_F4n1_GR_256X256 --name CHECK_F4_veldiv_256X256_scale_minus1600_900 --model bicycle_gan --direction AtoB --batch_size 2 --input_nc 1 --output_nc 1 --netG unet_256 --netD basic_256_multi --netD2 basic_256_multi --load_size 256 --crop_size 256 --num_threads=4 --norm instance --display_id 0 --lr 2e-4 --save_latest_freq 800000 --save_epoch_freq 5 --preprocess none --ndf 128 --ngf 128 --nef 128 --display_freq 10000000 --phase train --netE resnet_256 --gan_mode lsgan --nz 128 --lr_policy step --lr_decay_iters 25 --niter 0 --niter_decay 75 --where_add all --use_dropout --lambda_L1 20.0 --lambda_GAN 2.0

# --netD2 basic_512_multi
# See https://github.com/junyanz/BicycleGAN/issues/29#issuecomment-402540477

