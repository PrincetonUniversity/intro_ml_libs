# NVIDIA Rapids

[Rapids](https://rapids.ai/start.html) is something like Pandas, Scikit-learn and NetworkX (and more) running on GPUs:

+ `cuDF` is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data with a Pandas-like API.

+ `cuML` is a suite of GPU libraries that implement machine learning algorithms that share compatible APIs with other RAPIDS projects and Scikit-learn.

+ `cuGraph` is a GPU accelerated graph analytics library with functionality like `NetworkX` 

+ See also `cuSignal`, `cuXFilter`, `cuSpatial` and more

+ Multiple GPUs can be used when lots of processing must be done or if more GPU memory is needed

+ Spark 3.0 integrates with Rapids

<p align="center"><img src="https://github.com/rapidsai/cudf/blob/branch-0.13/img/rapids_arrow.png" width="80%"/></p>

## Installation

### Adroit or Tiger

Install `cuml` and its dependencies `cudf` and `dask-cudf`:

```bash
# for live workshop ~/.condarc should be directing the install to /scratch/network or /scratch/gpfs
$ module load anaconda3/2020.11
$ conda create -n rapids-0.18 -c rapidsai -c nvidia -c conda-forge -c defaults cuml=0.18 python=3.8 cudatoolkit=11.0
```

Or install all components of Rapids:

```bash
# for live workshop ~/.condarc should be directing the install to /scratch/network or /scratch/gpfs
$ module load anaconda3/2020.11
$ conda create -n rapids-0.18 -c rapidsai -c nvidia -c conda-forge -c defaults rapids=0.18 python=3.8 cudatoolkit=11.0
```

There is also a container [here](https://hub.docker.com/r/rapidsai/rapidsai/) which can be used on our systems with [Singularity](https://researchcomputing.princeton.edu/support/knowledge-base/singularity).


### Traverse

`cuDF` and `cuML` are available in the IBM WML-CE channel. You can make an environment like this:

```
$ ssh <YourNetID>@traverse.princeton.edu
$ module load anaconda3/2020.11
$ CHNL="https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda"
$ conda create --name rapids-env --channel ${CHNL} cudf cuml
# accept the license agreement
```

There are also dask-based packages available like dask-cudf.

## Using cuDF

See this [guide](https://docs.rapids.ai/api/cudf/stable/) for a 10-minute introduction to `cuDF` and `Dask-cuDF`.

Note that Rapids requires a GPU with compute capability (CC) of 6.0 and greater. This means the K40c GPUs on `adroit-h11g4` cannot be used (they are CC 3.5). On Adroit we mut request a V100 GPU (CC 7.0). TigerGPU is CC 6.0.

Below is a simple interactive session on Adroit checking the installation:

```bash
$ salloc -N 1 -n 1 -t 5 --gres=gpu:tesla_v100:1
$ module load anaconda3/2020.11
$ conda activate rapids-0.16
$ python
>>> import cudf
>>> s = cudf.Series([1, 2, 3, None, 4])
>>> s
0       1
1       2
2       3
3    <NA>
4       4
dtype: int64
>>> exit()
$ exit
```

Use `#SBATCH --gres=gpu:tesla_v100:1` on Adroit and `#SBATCH --gres=gpu:1` on Tiger.


Submitting a job to the Slurm scheduler:

```bash
#!/bin/bash
#SBATCH --job-name=rapids        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # this line is for adroit only

module purge
module load anaconda3/2020.11
conda activate rapids-0.16

python rapids.py
```

```python
import cudf
s = cudf.Series([1, 2, 3, None, 4])
print(s)
```

The output is

```
0       1
1       2
2       3
3    <NA>
4       4
dtype: int64
```

## cuDF with Multiple GPUs

```python
import cudf
import dask_cudf

df = cudf.DataFrame({'a':list(range(20, 40)), 'b':list(range(20))})

ddf = dask_cudf.from_cudf(df, npartitions=2)
print(ddf.compute())
```

Below is an appropriate Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=rapids        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2

module purge
module load anaconda3/2020.11
conda activate rapids-0.16

python rapids.py
```

Below is the output of this simple example:

```
     a   b
0   20   0
1   21   1
2   22   2
3   23   3
4   24   4
5   25   5
6   26   6
7   27   7
8   28   8
9   29   9
10  30  10
11  31  11
12  32  12
13  33  13
14  34  14
15  35  15
16  36  16
17  37  17
18  38  18
19  39  19
```

## Machine Learning with cuML

See the [documentation](https://docs.rapids.ai/api/cuml/stable/) for `cuML` and example Jupyter [notebooks](https://github.com/rapidsai/cuml/tree/branch-0.17/notebooks).

Follow the commands below to run a simple example on Adroit:

```python
import cudf
from cuml.linear_model import LogisticRegression
import numpy as np

N = 10000
X = np.random.normal(size=(N, 10)).astype(np.float32)
y = np.asarray([0,1]*(N//2), dtype=np.int32)

reg = LogisticRegression()
reg.fit(X, y)

X_test = np.random.normal(size=(10, 10)).astype(np.float32)
print(reg.predict(X_test))
```

## Useful Links

[NVIDIA Rapids on GitHub](https://github.com/rapidsai)
