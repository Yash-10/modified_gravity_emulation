#!/bin/bash -l

#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH -J ml_mg_train
#SBATCH -o standard_output_file.%J.out
#SBATCH -e standard_error_file.%J.err
#SBATCH -p cosma8-shm
#SBATCH -A durham
#SBATCH -t 48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=f20190481@goa.bits-pilani.ac.in

#load the modules used to build your program.
#module purge
#module load cuda/11.2
module unload python/3.6.5
module load python/3.8.7-C8

python3 plot_density_histogram.py
