#!/bin/bash
module purge
module load anaconda3/2020.7
conda create --name jax-gpu numpy scipy cython six -y
conda activate jax-gpu

git clone https://github.com/google/jax
cd jax
module load cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5 rh/devtoolset/8
python build/build.py --enable_cuda  \
                      --cudnn_path /usr/local/cudnn/cuda-10.2/7.6.5 \
                      --cuda_compute_capabilities 6.0 \
                      --enable_march_native \
                      --enable_mkl_dnn
pip install -e build
pip install -e .
