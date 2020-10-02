#!/bin/bash
#SBATCH -J platipus           # Job name
#SBATCH -o platipus.o%j       # Name of stdout output file
#SBATCH -e platipus.e%j       # Name of stderr error file
#SBATCH -p v100	          # Queue (partition) name
#SBATCH -N 5              # Total # of nodes 
#SBATCH -n 20              # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


module load python3
module load launcher_gpu
export LAUNCHER_WORKDIR=$SCRATCH/platipus
export LAUNCHER_JOB_FILE=parametric_longhorn1

$LAUNCHER_DIR/paramrun