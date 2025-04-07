#!/bin/bash --login

#SBATCH --job-name=test_sweep_vshort
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:0:0

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=n01-SOWISE
#SBATCH --partition=standard
#SBATCH --qos=standard

echo print python path from python
python print_path.py

echo path
echo $PATH

echo pythonpath
echo $PYTHONPATH

module load PrgEnv-gnu/8.3.3
module load cray-parallel-netcdf/1.12.3.1
module load cray-hdf5-parallel/1.12.2.1

echo pythonpath
echo $PYTHONPATH

work=/mnt/lustre/a2fs-work1/work/n01/n01/sithom

source $work/.bashrc

echo "hook 1"

eval "$($work/miniconda3/bin/conda shell.bash hook)"

echo which python
which python
echo "python --version"
python --version

echo "hook 2"

eval "$(conda shell.bash hook)"

echo which python
which python
echo "python --version"
python --version

echo print python path from python
python print_path.py


echo which conda
which conda
echo "conda --version"
conda --version

echo path
echo $PATH

echo "source"
# $work/miniconda3/bin/conda init bash
source activate $work/adcirc-emulation/env

echo which python
which python
echo "python --version"
python --version

# source $work/adcirc-emulation/env/bin/activate
conda activate $work/adcirc-emulation/env
# $work/miniconda3/bin/conda activate $work/adcirc-emulation/env 
# /work/n01/n01/sithom/miniconda3/condabin/conda activate /work/n01/n01/sithom/adcirc-emulation/env
echo path
echo $PATH


echo zeroth python run
python $work/adcirc-emulation/ademu/models/emu6d.py
python=$work/adcirc-emulation/env/bin/python

echo pythonpath
echo $PYTHONPATH
PYTHONPATH=$work/adcirc-emulation/env/bin/
echo pythonpath
echo $PYTHONPATH

echo path
echo $PATH

echo which python
which python
echo "python --version"
python --version

echo first python run
python $work/adcirc-emulation/ademu/models/emu6d.py
# python=$work/adcirc-emulation/env/bin/python


# srun launches the parallel program based on the SBATCH options
# srun 
echo "run python"
$work/adcirc-emulation/env/bin/python $work/adcirc-emulation/ademu/models/emu6d.py

echo "run python2"
eval "($work/adcirc-emulation/env/bin/python $work/adcirc-emulation/ademu/models/emu6d.py)"

echo "run conda python "
conda run -p $work/adcirc-emulation/env $work/adcirc-emulation/env/bin/python $work/adcirc-emulation/ademu/models/emu6d.py

echo "second conda run"
eval "($work/miniconda3/bin/conda run -p $work/adcirc-emulation/env $work/adcirc-emulation/env/bin/python $work/adcirc-emulation/ademu/models/emu6d.py)"

echo "third conda run"
eval "($work/miniconda3/bin/conda run -p $work/adcirc-emulation/env python $work/adcirc-emulation/ademu/models/emu6d.py)"


echo "fourth conda run"
eval "(conda run -p $work/adcirc-emulation/env python $work/adcirc-emulation/ademu/models/emu6d.py)"

echo path
echo $PATH
echo PYTHONPATH
echo $PYTHONPATH
echo $PYTHONUSERPATH

echo "try srun python"
srun python $work/adcirc-emulation/ademu/models/emu6d.py

srun python --version
srun which python

unset PYTHONPATH
python3.8 $work/adcirc-emulation/ademu/models/emu6d.py

python3.8 print_path.py