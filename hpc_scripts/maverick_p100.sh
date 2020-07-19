#!/bin/bash
#SBATCH -J platipus_p100           # Job name
#SBATCH -o platipus_p100.o%j       # Name of stdout output file
#SBATCH -e platipus_p100.e%j       # Name of stderr error file
#SBATCH -p p100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes 
#SBATCH -n 3              # Total # of mpi tasks
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=vshekar@haverford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


# Other commands must follow all #SBATCH directives...




# Launch MPI code... 

#mpirun python3 run_mpi.py         # Use ibrun instead of mpirun or mpiexec
module load python3
ibrun -n 3 python3 -m hpc_scripts.run_mpi_p100


