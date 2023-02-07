#!/bin/bash --login

#SBATCH --job-name=test_sweep_vshort
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=1:0:0

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=n01-SOWISE
#SBATCH --partition=standard
#SBATCH --qos=standard

# Load the xthi module to get access to the xthi program
# module load 
module load PrgEnv-gnu/8.0.0
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel


source ~/.bashrc
source activate ~/sithom/new-orleans/env
~/miniconda3/condabin/conda activate /work/n01/n01/sithom/new-orleans/env

which python

# srun launches the parallel program based on the SBATCH options
srun python /work/n01/n01/sithom/new-orleans/src/models/emu6d.py

