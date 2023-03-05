#!/bin/sh
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH -J testing_F6_VELDIV
#SBATCH -o testing_standard_output_file.%J.out
#SBATCH -e testing_standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=f20190481@goa.bits-pilani.ac.in

module unload python/3.6.5
module load python/3.8.7-C8

#python3 -u testing.py > output.txt
python3 -u testing.py
