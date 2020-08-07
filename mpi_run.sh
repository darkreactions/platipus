#!/bin/bash
#SBATCH -J al_svm           # Job name
#SBATCH -o al_svm.o%j       # Name of stdout output file
#SBATCH -e al_svm.e%j       # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 3              # Total # of mpi tasks
#SBATCH -t 04:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


# Other commands must follow all #SBATCH directives...
module load python3
module list
pwd
date

# Launch MPI code... 

ibrun python3 run_mpi.py         # Use ibrun instead of mpirun or mpiexec
