#!/bin/bash --login

#SBATCH --job-name=active_test
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=3
#SBATCH --time=24:0:0

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

# try to activate bashrc

work=/mnt/lustre/a2fs-work1/work/n01/n01/sithom

source $work/.bashrc


echo "hook 1"
eval "$(conda shell.bash hook)"

echo which python
which python
echo "python --version"
python --version

# python new-orleans/src/models/emu6d.py samples=100 seed=31 dryrun=false
python new-orleans/src/models/emu6d.py init_samples=29 active_samples=1 seed=40 dryrun=false
