#!/bin/bash
#SBATCH --job-name=flux-cpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)

module purge
module load julia/1.4.1

julia ../60-minute-blitz.jl
