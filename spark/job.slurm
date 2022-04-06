#!/bin/bash
#SBATCH --job-name=spark-ml      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=spark

module purge
module load anaconda3/2021.11
module load spark/hadoop3.2/3.2.0

spark-start
spark-submit --total-executor-cores 6 --executor-memory 4G random_forest_classifier_example.py
