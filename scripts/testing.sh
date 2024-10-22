#!/bin/sh
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH -J testing_F6_VELDIV
#SBATCH -o testing_standard_output_file.%J.out
#SBATCH -e testing_standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 02:00:00

module unload python/3.6.5
module load python/3.8.7-C8

#python3 -u testing.py > output.txt
#python3 -u testing_for_den_F4den_validation.py
#python3 -u testing_for_den_F5den_validation.py
#python3 -u testing_for_den_F6den_validation.py
#python3 -u testing_for_den_F4den_test.py
#python3 -u testing_for_den_F5den_test.py
#python3 -u testing_for_den_F6den_test.py
#python3 -u testing_for_den_F4den_validation_Bicycle.py
#python3 -u testing_for_veldiv_F4den_validation_Bicycle.py
#python3 -u testing_for_den_F4den_test_Bicycle.py
#python3 -u testing_for_den_F4veldiv_test_Bicycle.py

#python3 -u testing_for_den_F4den_validation_Bicycle_latent_interp.py
#python3 -u testing_for_den_F4den_test_Bicycle_latent_interp.py
python3 -u testing_for_den_F4veldiv_test_Bicycle_latent_interp.py

