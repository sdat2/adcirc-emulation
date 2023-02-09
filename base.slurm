#!/bin/bash --login

#SBATCH --job-name=multi_sweep_vshort
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=10
#SBATCH --time=2:0:0

# Replace [budget code] below with your project code (e.g. t01)

#SBATCH --account=n01-SOWISE
#SBATCH --partition=standard
#SBATCH --qos=standard

# emailing for start and end.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=sdat2@cam.ac.uk

# Load modules
module load PrgEnv-gnu/8.0.0
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
# module load cray-python

echo pythonpath
echo $PYTHONPATH

work=/mnt/lustre/a2fs-work1/work/n01/n01/sithom

source $work/.bashrc

echo "hook 1"

#eval "$($work/miniconda3/bin/conda shell.bash hook)"

#echo which python
#which python
#echo "python --version"
#python --version

# echo "hook 2"

eval "$(conda shell.bash hook)"

echo which python
which python
echo "python --version"
python --version

WANDB_CACHE_DIR="/work/n01/n01/sithom/tmp"
WANDB_CONFIG_DIR="/work/n01/n01/sithom/.config/wandb"
MPLCONFIGDIR="/work/n01/n01/sithom/.config/matplotlib"

echo print python path from python
python print_path.py
python new-orleans/src/models/emu6d.py
