#!/bin/bash
#SBATCH -J Parametric_test
#SBATCH -N 4
#SBATCH -n 16
#SBATCH -p v100
#SBATCH -o Parametric.o%j
#SBATCH -e Parametric.e%j
#SBATCH -t 24:00:00
#------------------------------------------------------
module load launcher_gpu
module load python3
export LAUNCHER_WORKDIR=$SCRATCH/platipus
export LAUNCHER_JOB_FILE=parametric

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM


$LAUNCHER_DIR/paramrun
