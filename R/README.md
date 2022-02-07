# R

See [this page](https://cran.r-project.org/web/views/MachineLearning.html) for a list of ML packages on CRAN. For general information about using R on the HPC clusters see [this page](https://researchcomputing.princeton.edu/R). There is some information about running in parallel in [this repo](https://github.com/PrincetonUniversity/HPC_R_Workshop).

## Caret

The Caret package provides many ML models with the ability to train in parallel. In the example below we show how to train an RF model in parallel. Begin by installing the needed packages:

```
$ ssh <YourNetID>@adroit.princeton.edu
$ R
> install.packages(c("caret", "doParallel"))
> q()
```

Below is the R script:

```R
library(caret)
credit <- read.csv("credit.csv")

# train in serial
system.time(train(default ~ ., data = credit, method="rf"))

# train in parallel
cpucores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
library(doParallel)
registerDoParallel(cores=cpucores)
system.time(train(default ~ ., data = credit, method="rf"))
```

Below is an appropriate Slurm script that takes advantage of the parallelism offered by `caret`:

```bash
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
Rscript demo.R
```

Submit the job:

```
$ cd intro_ml_libs/R
$ wget https://raw.githubusercontent.com/PacktPublishing/Machine-Learning-with-R-Third-Edition/master/Chapter05/credit.csv
$ sbatch job.slurm
```

When the job completes the output will be:

```
$ cat slurm-xxxxxx.out
Loading required package: lattice
Loading required package: ggplot2
   user  system elapsed 
102.538   0.461 103.014 
Loading required package: foreach
Loading required package: iterators
Loading required package: parallel
   user  system elapsed 
104.860   1.395  29.412
```

We see that serial training requires 103 seconds while when four CPU-cores are used in parallel the time is reduced to 29 seconds. An RF model is composed of many independent decision trees. In this case the speed-up arises from the trees being trained in parallel.

## XGBoost

The `xgboost` package is [available](https://xgboost.readthedocs.io/en/latest/R-package/index.html) for R. It can be used on [GPUs](https://xgboost.readthedocs.io/en/latest/build.html#installing-r-package-with-gpu-support).

## R Interface to TensorFlow

See an overview of the procedure here: [R Interface to TensorFlow](https://tensorflow.rstudio.com/). Below are directions for getting this work on Adroit:

```
$ ssh adroit
$ module load anaconda3
$ pip install --user virtualenv
$ cd software
$ mkdir R-tf2-env
$ virtualenv R-tf2-env
$ source R-tf2-env/bin/activate
$ pip install tensorflow-gpu h5py pyyaml requests Pillow scipy
$ deactivate

$ module load rh/devtoolset/8
$ R
> install.packages("tensorflow")
> install.packages("keras")
```

This was worked out in CSES ticket 32139.

Start of main.R will look like:

```
library(tensorflow)
use_python("/home/aturing/software/R-tf2-env/bin/python")
use_virtualenv("/home/aturing/software/R-tf2-env")
library(keras)
```

## MXNet

 >  [MXNet](https://mxnet.apache.org/api/r) supports the R programming language. The MXNet R package brings flexible and efficient GPU computing and state-of-art deep learning to R. It enables you to write seamless tensor/matrix computation with multiple GPUs in R. It also lets you construct and customize the state-of-art deep learning models in R, and apply them to tasks, such as image classification and data science challenges.

