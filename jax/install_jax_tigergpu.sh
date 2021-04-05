#!/bin/bash

module purge
module load anaconda3/2020.11
conda create --name jax-gpu python=3.8 numpy scipy six wheel -y
conda activate jax-gpu

export TMP=/tmp

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2 rh/devtoolset/8

# install jaxlib
python build/build.py --enable_cuda  \
                      --cuda_path /usr/local/cuda-11.0 \
                      --cudnn_path /usr/local/cudnn/cuda-11.0/8.0.2 \
                      --cuda_compute_capabilities 6.0 \
                      --target_cpu_features native \
                      --enable_mkl_dnn
pip install dist/*.whl

# install jax
pip install -e .
