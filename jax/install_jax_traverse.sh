#!/bin/bash

# JAX is unsupported on the POWER architecture. These directions may or may not work.
# You may consider using a stable release instead of the master branch on github.

module purge
module load anaconda3/2023.3
conda create --name jax-gpu python=3.9 numpy scipy six wheel bazel --channel conda-forge -y
conda activate jax-gpu

export TMP=/tmp

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0

# install jaxlib
python build/build.py --enable_cuda \
                      --cuda_path /usr/local/cuda-11.7 \
                      --cudnn_path /usr/local/cudnn/cuda-11.3/8.2.0 \
                      --noenable_mkl_dnn \
                      --cuda_compute_capabilities 7.0
pip install dist/*.whl

# install jax
pip install -e .
