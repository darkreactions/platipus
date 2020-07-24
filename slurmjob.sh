#!/bin/bash
#SBATCH -J Parametric_test
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p development
#SBATCH -o Parametric.o%j
#SBATCH -e Parametric.e%j
#SBATCH -t 00:05:00
#          <------ Account String ----->
# <--- (Use this ONLY if you have MULTIPLE accounts) --->
##SBATCH -A
#------------------------------------------------------
module load launcher_gpu
module load python3
export LAUNCHER_WORKDIR=$SCRATCH/platipus
export LAUNCHER_JOB_FILE=parametric

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM


$LAUNCHER_DIR/paramrun
