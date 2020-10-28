#!/bin/bash
#SBATCH --job-name=spark-ml      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # memory per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2018.12 spark/hadoop2.7/2.4.6

spark-start
spark-submit --total-executor-cores 6 --executor-memory 4G random_forest_classifier_example.py
