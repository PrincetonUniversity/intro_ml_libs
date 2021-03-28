# JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research.

## Installation

### GPU Version (TigerGPU, Traverse, Adroit)

JAX must be built from source to use on the GPU clusters as [described here](https://jax.readthedocs.io/en/latest/developer.html). Below is the build procedure for TigerGPU (for Traverse and Adroit see notes below and use either `install_jax_traverse.sh` or `install_jax_adroit.sh`):

```
$ ssh <YourNetID>@tigergpu.princeton.edu
$ cd software  # or another directory
$ wget https://raw.githubusercontent.com/PrincetonUniversity/intro_ml_libs/master/jax/install_jax_tigergpu.sh
$ bash install_jax_tigergpu.sh | tee jax.log
```

For Traverse and Adroit, use `--cuda_compute_capabilities 7.0` instead of 6.0. You also may need to use different modules. On Traverse it may be necessary to use stable releases instead of the master branch on github. JAX does not formally support the POWER architecture.

If you do a pip install instead of building from source on TigerGPU then you will encounter the following error when you try to import jax:

```
ImportError: /lib64/libm.so.6: version `GLIBC_2.23' not found
```

Follow the directions above to build from source.

### CPU-Only Version (Della, Perseus)

Here are the installation directions for the CPU-only clusters:

```
$ module load anaconda3/2020.11
$ conda create --name jax-cpu --channel conda-forge --override-channels jax "libblas=*=*mkl"
```

See [this page](https://researchcomputing.princeton.edu/python) for Slurm scripts. Be sure to take advantage of the parallelism of the CPU version which uses MKL and OpenMP. For the MNIST example, one finds as `cpus-per-task` increases from 1, 2, 4, the run time decreases as 139 s, 87 s, 58 s.

```
$ cd /scratch/gpfs/<YourNetID>
$ mkdir myjob && cd myjob
$ wget https://raw.githubusercontent.com/PrincetonUniversity/intro_ml_libs/master/jax/download_data.py
$ python3 download_mnist.py
```

## Example Job for GPU Version

First obtain the JAX script. Also, the compute nodes do not have internet access so we have to download the data on the head node:

```bash
$ ssh <YourNetID>@tigergpu.princeton.edu
$ module load anaconda3/2020.11
$ conda activate jax-gpu  # installation directions above
$ mkdir /scratch/gpfs/<YourNetID>/jax_test && cd /scratch/gpfs/<YourNetID>/jax_test
$ wget https://raw.githubusercontent.com/google/jax/master/examples/mnist_classifier.py
$ wget <download_data.py>
$ python download_data.py
```

The JAX source code needs to be modified so that it doesn't try to perform the download on the compute node. With the installation having been done in `~/software`, next you need to make the `mnist_raw()` function in `software/jax/examples/datasets.py` look like this:

```python
#for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
#                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
#    _download(base_url + filename, filename)
 
  _DATA = os.getcwd() + "/data"
  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))
```

The Slurm script below (job.slurm) may be used on Tiger when JAX is built according to `install_jax_tigergpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=jax-gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2020.7 cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5
conda activate jax-gpu
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-10.2

python mnist_classifier.py
```

On adroit use the modules used in `install_jax_adroitgpu.sh`. Submit the job with the following command:

```
$ sbatch job.slurm
```

Here is the output of the job:

```
Starting training...
Epoch 0 in 2.20 sec
Training set accuracy 0.8719333410263062
Test set accuracy 0.8804000616073608
Epoch 1 in 0.42 sec
Training set accuracy 0.8979166746139526
Test set accuracy 0.9032000303268433
Epoch 2 in 0.42 sec
Training set accuracy 0.9092666506767273
Test set accuracy 0.9143000245094299
Epoch 3 in 0.42 sec
Training set accuracy 0.9170666933059692
Test set accuracy 0.9221000671386719
Epoch 4 in 0.42 sec
Training set accuracy 0.9226500391960144
Test set accuracy 0.9280000329017639
Epoch 5 in 0.42 sec
Training set accuracy 0.9271833300590515
Test set accuracy 0.929900050163269
Epoch 6 in 0.42 sec
Training set accuracy 0.932366669178009
Test set accuracy 0.932900071144104
Epoch 7 in 0.42 sec
Training set accuracy 0.9357333183288574
Test set accuracy 0.9364000558853149
Epoch 8 in 0.42 sec
Training set accuracy 0.938800036907196
Test set accuracy 0.9394000172615051
Epoch 9 in 0.42 sec
Training set accuracy 0.9425666928291321
Test set accuracy 0.9419000744819641
```
