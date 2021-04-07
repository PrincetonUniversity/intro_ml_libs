#!/bin/bash
#SBATCH --job-name=flux-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node (ADROIT ONLY)

module purge
module load julia/1.5.0 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2

julia conv_mnist.jl
