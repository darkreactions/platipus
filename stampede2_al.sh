#!/bin/bash
#SBATCH -J platipus           # Job name
#SBATCH -o platipus.o%j       # Name of stdout output file
#SBATCH -e platipus.e%j       # Name of stderr error file
#SBATCH -p skx-normal 	          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 48             # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


module load python3
module load launcher
export LAUNCHER_WORKDIR=$WORK/platipus
export LAUNCHER_JOB_FILE=al_ft_parametric

$LAUNCHER_DIR/paramrun