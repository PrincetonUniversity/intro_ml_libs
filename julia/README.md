# Julia

According to the [Julia](https://docs.julialang.org/en/v1/) website:

> [Julia] is a flexible dynamic language, appropriate for scientific and numerical computing, with performance comparable to traditional statically-typed languages. Once you understand how Julia works, it's easy to write code that's nearly as fast as C. Julia features optional typing, [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), and good performance, achieved using type inference and just-in-time (JIT) compilation, implemented using LLVM. It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.

Two popular machine learning packages for Julia are Flux and Knet.

## Flux with CPUs

Here is a 60-minute [introduction](https://github.com/FluxML/model-zoo/blob/master/tutorials/60-minute-blitz.jl) to Flux.

```bash
$ ssh adroit
$ module load julia/1.4.1
$ julia
julia> ]
(@v1.4) pkg> add Flux, Zygote, Metalhead, Images
# press the backspace or delete key
julia> using Metalhead
julia> Metalhead.download(CIFAR10)  # download data since no internet access on compute nodes
julia> exit()
```

Now run the script:

```bash
$ cd intro_machine_learning_libs/julia/flux_cpu
$ wget https://raw.githubusercontent.com/FluxML/model-zoo/master/tutorials/60-minute-blitz.jl
# comment out line 287 since we already downloaded the images, i.e.,  #Metalhead.download(CIFAR10)
$ sbatch job.slurm  # this will take about 50 minutes to run
```

Here is the output:

```
WARNING: using Images.data in module Main conflicts with an existing identifier.
AbstractFloat[0.023375249260704556 0.021599333216003345 0.006813448722365639 0.019994560635509962 0.02818629433346919 0.00523279341688931 0.024854965124418937 0.028238140104154912 0.025300470349564185 0.027094133068334553; 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0; 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0 0.0f0; 0.0013235271222324005 0.0012229731976194532 0.00038578344467753563 0.0011321095661086483 0.0015959327153907523 0.0002962853544383299 0.0014073099327163 0.0015988682684156756 0.0014325348294435926 0.0015340935863129997; -0.012773237532978408 -0.01180280093037371 -0.0037231602529196467 -0.010925884447968043 -0.015402198653816345 -0.002859422482706746 -0.013581817668220583 -0.015430529403861536 -0.013825260807564107 -0.014805394953103103]
AbstractFloat[0.03053741375792813, 0.0f0, 0.0f0, 0.0017290551600403561, -0.016686951023387797]
[-0.008607726965518462 -0.0 -0.0 -0.010647063898407883 -0.012242528398697909; 0.008607736541397593 0.0 0.0 0.010647075742997912 0.012242542018201952]
[-0.026789205376463534, 0.026789235178778795]
accuracy(valX, valY) = 0.135
accuracy(valX, valY) = 0.198
accuracy(valX, valY) = 0.291
accuracy(valX, valY) = 0.327
accuracy(valX, valY) = 0.343
accuracy(valX, valY) = 0.391
accuracy(valX, valY) = 0.389
accuracy(valX, valY) = 0.437
accuracy(valX, valY) = 0.473
accuracy(valX, valY) = 0.488
```

The script took 50 minutes to run and required 2.4 GB of memory.

## Flux with GPUs

First we need to add the GPU packages:

```bash
# ssh adroit
$ module load julia/1.5.0 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2
$ julia
julia> ]
(v1.2) pkg> add Flux, Zygote, Metalhead, Images, CUDA
$ # press the backspace or delete key
julia> exit()
```



```bash
$ cd intro_machine_learning_libs/julia/flux_gpu
$ wget https://raw.githubusercontent.com/FluxML/model-zoo/master/tutorials/60-minute-blitz.jl
# comment out line 287 since we already downloaded the images, i.e.,  #Metalhead.download(CIFAR10)
# uncomment line 282 to use the GPU, i.e., CUDA
$ julia
julia> using Metalhead
julia> Metalhead.download(CIFAR10)  # download data since no internet access on compute nodes
julia> exit()
$ sbatch job.slurm
```

Below is an appropriate Slurm script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=flux-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load julia/1.5.0 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2

julia ../60-minute-blitz.jl
```

Submit the job with: sbatch job.slurm

Here's the output on Adroit:

```
┌ Warning: Some registries failed to update:
│     — /home/jdh4/.julia/registries/General — failed to fetch from repo
└ @ Pkg.Types /buildworker/worker/package_linux64/build/usr/share/julia/stdlib/v1.2/Pkg/src/Types.jl:1171
[ Info: Building the CUDAnative run-time library for your sm_35 device, this might take a while...
Activating environment at `~/flux-env/Project.toml`
  Updating registry at `~/.julia/registries/General`
  Updating git-repo `https://github.com/JuliaRegistries/General.git`
Float32[2.0, 4.0, 6.0]
Float32[-0.835879, 0.3685953, 1.0108142, -0.29181987, 0.31272212]
```

The GPU version ran about 2.6x faster than the CPU version.

Note that there are no GPUs on the head node of TigerGPU and no internet connection on the compute nodes. To run MNIST for example you will need to download the data first.

## Knet


## TensorFlow

There is a Julia package called [TensorFlow.jl](https://github.com/malmaud/TensorFlow.jl) that provides an interface to TensorFlow. It can be used with up to version 1.12 of TensorFlow. It appears that the number of commits is decreasing with time on this repo.

## Conventional Models

[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) This using PyCall  
[Machine Learning Toolbox for Julia](https://github.com/alan-turing-institute/MLJ.jl) MLJ is native Julia (it include ScikitLearn.jl)  
[CombineML](https://github.com/ppalmes/CombineML.jl) CombineML is a heterogeneous ensemble learning package for Julia

## More Info

[Julia Documentation](https://docs.julialang.org/en/v1/)  
[Flux website](https://fluxml.ai/)  
[Flux documentation](https://fluxml.ai/Flux.jl/stable/)  
[Flux on GitHub](https://github.com/FluxML/Flux.jl)  
[Knet documentation](https://denizyuret.github.io/Knet.jl/latest/)  
[Knet on GitHub](https://github.com/denizyuret/Knet.jl)  
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)

