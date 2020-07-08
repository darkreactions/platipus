#!/bin/bash
#SBATCH -J platipus           # Job name
#SBATCH -o platipus.o%j       # Name of stdout output file
#SBATCH -e platipus.e%j       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 4              # Total # of mpi tasks
#SBATCH -t 8:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


# Other commands must follow all #SBATCH directives...




# Launch MPI code... 

#mpirun python3 run_mpi.py         # Use ibrun instead of mpirun or mpiexec
ibrun --multi-prog


