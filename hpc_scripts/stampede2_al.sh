#!/bin/bash
#SBATCH -J platipus           # Job name
#SBATCH -o platipus.o%j       # Name of stdout output file
#SBATCH -e platipus.e%j       # Name of stderr error file
#SBATCH -p normal 	          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 15             # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

module load gcc
module load python3
module load launcher
export LAUNCHER_WORKDIR=/work/06453/vshekar/stampede2/platipus
export LAUNCHER_JOB_FILE=segfault_params

$LAUNCHER_DIR/paramrun