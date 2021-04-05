#!/bin/bash

# JAX is unsupported on the POWER architecture. These directions may or may not work.
# You may consider using a stable release instead of the master branch on github.

module purge
module load anaconda3/2020.11
conda create --name jax-gpu python=3.7 numpy scipy six wheel -y
conda activate jax-gpu

export TMP=/tmp

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.1 cudnn/cuda-11.1/8.0.4

# install jaxlib
python build/build.py --enable_cuda \
                      --cuda_path /usr/local/cuda-11.1 \
                      --cudnn_path /usr/local/cudnn/cuda-11.1/8.0.4 \
                      --noenable_mkl_dnn \
                      --cuda_compute_capabilities 7.0 \
                      --bazel_path /usr/bin/bazel
pip install dist/*.whl

# install jax
pip install -e .
