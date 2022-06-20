# JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research. JAX can be used for:

- automatic differentiation of Python and NumPy functions (more general then TensorFlow)
- a good choice for non-conventional neural network architectures and loss functions
- accelerating code using a JIT
- carrying out computations using multiple GPUs/TPUs

## Conda Installation

The easiest way to install JAX is witih conda:

```
$ module load anaconda3/2021.11
$ CONDA_OVERRIDE_CUDA="11.2" conda create --name jax-env jax "jaxlib==0.3.10=cuda112*" -c conda-forge
```

The directions above are for jaxlib version 0.3.10 with CUDA 11.2. To see the latest version use this command:

```
$ module load anaconda3/2021.11
$ conda search jaxlib -c conda-forge
...
jaxlib                        0.3.10 cuda112py39h8d07533_0  conda-forge
```

## Pip Installation

### Della

Run the commands below to install `jax` on Della:

```
$ ssh <YourNetID>@della-gpu.princeton.edu
$ module load anaconda3/2021.11
$ conda create --name jax-gpu python=3.9 matplotlib
$ conda activate jax-gpu
$ pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Since `della-gpu` has a GPU, we can run a short test:

$ module load cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0
$ python
>>> import jax.numpy as jnp
>>> jnp.arange(3)
DeviceArray([0, 1, 2], dtype=int32)
```

The correct environment modules to use in the Slurm script are `anaconda3/2021.11`, `cudatoolkit/11.7` and `cudnn/cuda-11.x/8.2.0`.

### TigerGPU

Run these commands to install `jax`:

```
$ module load anaconda3/2021.11
$ conda create --name jax-gpu python=3.9 matplotlib
$ conda activate jax-gpu
$ pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The correct environment modules to use in the Slurm script are `anaconda3/2021.11`, `cudatoolkit/11.3`, `cudnn/cuda-11.x/8.2.0` and `nvhpc/21.5`. The `nvhpc/21.5` module is needed to avoid the error: `ptxas returned an error during compilation of ptx to sass`

### Adroit (GPU)

Run these commands to install `jax`:

```
$ module load anaconda3/2021.11
$ conda create --name jax-gpu python=3.9 matplotlib
$ conda activate jax-gpu
$ pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

The correct environment modules to use in the Slurm script are `anaconda3/2021.11`, `cudatoolkit/11.7` and `cudnn/cuda-11.5/8.3.2`.

### Traverse

JAX is not supported on the POWER architecture. In the past we have found it to install and run successfully but this is no longer the case. You can try the build script if you like:

```
$ ssh <YourNetID>@traverse.princeton.edu
$ cd software  # or another directory
$ wget https://raw.githubusercontent.com/PrincetonUniversity/intro_ml_libs/master/jax/install_jax_traverse.sh
$ bash install_jax_traverse.sh | tee jax.log
```

You will probably encounter the following error which arises because the build system produces an x86_64 wheel:

```
ERROR: jaxlib-0.1.65-cp37-none-manylinux2010_x86_64.whl is not a supported wheel on this platform.
```

### CPU-Only Version (Della, Perseus)

Here are the installation directions for the CPU-only clusters:

```
$ module load anaconda3/2021.11
$ conda create --name jax-cpu --channel conda-forge --override-channels jax "libblas=*=*mkl"
```

See [this page](https://researchcomputing.princeton.edu/python) for Slurm scripts. Be sure to take advantage of the parallelism of the CPU version which uses MKL and OpenMP. For the MNIST example, one finds as `cpus-per-task` increases from 1, 2, 4, the run time decreases as 139 s, 87 s, 58 s.

## Example Job for GPU Version

Run the commands below to submit the test job. Recall that the compute nodes do not have internet access so we have to download the data on the head node in advance.

```bash
$ ssh <YourNetID>@tigergpu.princeton.edu
$ module load anaconda3/2020.11
$ conda activate jax-gpu  # installation directions above
$ mkdir /scratch/gpfs/<YourNetID>/jax_test && cd /scratch/gpfs/<YourNetID>/jax_test
$ git clone https://github.com/google/jax
$ cd jax/examples
$ wget https://raw.githubusercontent.com/PrincetonUniversity/intro_ml_libs/master/jax/download_data.py
$ python download_data.py
```

Two files need to be modified. First, make the `mnist_raw()` function in `jax/examples/datasets.py` look like this:

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

Second, change `from examples import datasets` in `mnist_classify.py` to `import datasets`.

The Slurm script below (job.slurm) may be used on Tiger -- different modules are needed for Adroit:

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
module load anaconda3/2020.11 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2
conda activate jax-gpu

python mnist_classifier.py
```

Submit the job with the following command:

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
