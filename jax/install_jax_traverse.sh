#!/bin/bash

# JAX is unsupported on the POWER architecture. These directions may or may not work.
# You may consider using a stable release instead of the master branch on github.

module purge
module load anaconda3/2020.7
conda create --name jax-gpu python=3.7 numpy scipy cython six -y
conda activate jax-gpu

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.0 cudnn/cuda-11.0/8.0.1
python build/build.py --enable_cuda \
                      --cudnn_path /usr/local/cudnn/cuda-11.0/8.0.1 \
                      --noenable_march_native \
                      --noenable_mkl_dnn \
                      --cuda_compute_capabilities 7.0 \
                      --bazel_path /usr/bin/bazel
pip install -e build
pip install -e .
