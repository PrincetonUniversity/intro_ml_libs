#!/bin/bash

# ssh <YourNetID>@della-gpu.princeton.edu

module purge
module load anaconda3/2022.5
conda create --name jax-gpu-src python=3.9 numpy scipy six wheel -y
conda activate jax-gpu-src

export TMP=/tmp

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0

# install jaxlib
python build/build.py --enable_cuda  \
                      --cuda_path /usr/local/cuda-11.3 \
                      --cudnn_path /usr/local/cudnn/cuda-11.3/8.2.0 \
                      --cuda_compute_capabilities 8.0 \
                      --target_cpu_features native \
                      --enable_mkl_dnn
pip install dist/*.whl

# install jax
pip install -e .
