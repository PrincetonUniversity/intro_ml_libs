# Julia

According to the [Julia](https://docs.julialang.org/en/v1/) website:

> [Julia] is a flexible dynamic language, appropriate for scientific and numerical computing, with performance comparable to traditional statically-typed languages. Once you understand how Julia works, it's easy to write code that's nearly as fast as C. Julia features optional typing, [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), and good performance, achieved using type inference and just-in-time (JIT) compilation, implemented using LLVM. It is multi-paradigm, combining features of imperative, functional, and object-oriented programming.

Two popular deep learning packages for Julia are Flux and Knet.

## Flux

To learn about [Flux](https://fluxml.ai/Flux.jl/stable/) see this [60-minute blitz](https://github.com/FluxML/model-zoo/blob/master/tutorials/60-minute-blitz/60-minute-blitz.jl) and the [Flux Model Zoo](https://github.com/FluxML/model-zoo/).

### Flux Example Job with GPUs

Let's train a CNN on MNIST using this [Julia script](https://github.com/FluxML/model-zoo/blob/master/vision/conv_mnist/conv_mnist.jl). First we need to add the packages and download the data while on the login node:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ module load julia/1.5.0 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2
$ julia
julia> ]
(v1.5) pkg> add Flux, CUDA, TensorBoardLogger, ProgressMeter, BSON, MLDatasets
$ # press the backspace or delete key
julia> using MLDatasets
julia> MLDatasets.MNIST.download()  # answer y when prompted
julia> exit()
```

Download the Julia script and submit the job:

```
$ cd intro_machine_learning_libs/julia/flux_gpu
$ wget https://raw.githubusercontent.com/FluxML/model-zoo/master/vision/conv_mnist/conv_mnist.jl
$ sbatch job.slurm
```

The GPU usage is found to be fairly good fluctuating around 40-50%. The end of the output is:

```
[ Info: Model saved in "runs/model.bson"
Epoch: 0   Train: (loss = 2.3053f0, acc = 11.0917)   Test: (loss = 2.3057f0, acc = 11.28)
Epoch: 1   Train: (loss = 0.1958f0, acc = 94.1)   Test: (loss = 0.1811f0, acc = 94.77)
Epoch: 2   Train: (loss = 0.1181f0, acc = 96.4283)   Test: (loss = 0.1097f0, acc = 96.67)
Epoch: 3   Train: (loss = 0.0892f0, acc = 97.2483)   Test: (loss = 0.0846f0, acc = 97.43)
Epoch: 4   Train: (loss = 0.0735f0, acc = 97.7817)   Test: (loss = 0.0677f0, acc = 97.83)
Epoch: 5   Train: (loss = 0.0635f0, acc = 98.01)   Test: (loss = 0.0606f0, acc = 97.99)
Epoch: 6   Train: (loss = 0.0568f0, acc = 98.2433)   Test: (loss = 0.0552f0, acc = 98.19)
Epoch: 7   Train: (loss = 0.0503f0, acc = 98.4367)   Test: (loss = 0.0507f0, acc = 98.38)
Epoch: 8   Train: (loss = 0.0417f0, acc = 98.6967)   Test: (loss = 0.0444f0, acc = 98.58)
Epoch: 9   Train: (loss = 0.0436f0, acc = 98.595)   Test: (loss = 0.0471f0, acc = 98.41)
Epoch: 10   Train: (loss = 0.0363f0, acc = 98.905)   Test: (loss = 0.0408f0, acc = 98.7)
```

You many find harmless messages like `curl: (6) Could not resolve host: pkg.julialang.org; Unknown error` in the output which arise from the package manager calling curl from the compute nodes which do not have internet access.

#### Della

The procedure above can be used on Della (GPU) with the following modications:
- `ssh <YourNetID>@della-gpu.princeton.edu`
- `module load julia/1.7.1 cudatoolkit/11.6 cudnn/cuda-11.x/8.2.0`
- Run the job on the login node before submitting with sbatch since it needs to download additional software: `$ julia conv_mnist.jl`
- Then do a full production run by increasing epochs and use sbatch: `$ sbatch job.slurm`

## Knet

Knet is a deep learning framework implemented in Julia. It supports GPU operation and automatic differentiation using dynamic computational graphs for models defined in plain Julia.

[Knet documentation](https://denizyuret.github.io/Knet.jl/latest/)  
[Knet on GitHub](https://github.com/denizyuret/Knet.jl) 

## TensorFlow

There is a Julia package called [TensorFlow.jl](https://github.com/malmaud/TensorFlow.jl) that provides an interface to TensorFlow. It can be only be used with older versions of TensorFlow. The authors of this package recommend using Flux.

## Conventional Models

[ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl) This using PyCall  
[Machine Learning Toolbox for Julia](https://github.com/alan-turing-institute/MLJ.jl) MLJ is native Julia (it include ScikitLearn.jl)  
[CombineML](https://github.com/ppalmes/CombineML.jl) CombineML is a heterogeneous ensemble learning package for Julia

## More Info

[Julia Documentation](https://docs.julialang.org/en/v1/)  
[Flux website](https://fluxml.ai/)  
[Flux documentation](https://fluxml.ai/Flux.jl/stable/)  
[Flux on GitHub](https://github.com/FluxML/Flux.jl)   
[DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)

