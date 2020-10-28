# R

See [this page](https://cran.r-project.org/web/views/MachineLearning.html) for a list of ML packages on CRAN. For general information about using R on the HPC clusters see [this page](https://researchcomputing.princeton.edu/R). There is some information about running in parallel in [this repo](https://github.com/PrincetonUniversity/HPC_R_Workshop).

## Caret

Install the needed packages:

```
$ ssh <YourNetID>@adroit.princeton.edu
$ module load rh/devtoolset/8
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

```
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
$ cd intro_ml_libs/R/myjob
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

[documentation](https://xgboost.readthedocs.io/en/latest/R-package/index.html)

[GPU support](https://xgboost.readthedocs.io/en/latest/build.html#installing-r-package-with-gpu-support) is also available for XGBoost.

## R and Deep Learning

[MXNet](https://mxnet.apache.org/api/r)
 >  MXNet supports the R programming language. The MXNet R package brings flexible and efficient GPU computing and state-of-art deep learning to R. It enables you to write seamless tensor/matrix computation with multiple GPUs in R. It also lets you construct and customize the state-of-art deep learning models in R, and apply them to tasks, such as image classification and data science challenges.

[R Interface to TensorFlow](https://oncomputingwell.princeton.edu/2019/06/installing-and-using-tensorflow-with-r)

[R interface to keras](https://www.amazon.com/Deep-Learning-R-Francois-Chollet/dp/161729554X/ref=sr_1_3?keywords=deep+learning+with+R&qid=1583689546&sr=8-3)
