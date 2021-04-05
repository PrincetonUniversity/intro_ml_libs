# Intro to the Machine Learning Libraries

## Setup for live workshop

<!--+ Connect to the eduroam wireless network -->

+ Open a terminal (e.g., Terminal, PowerShell, PuTTY) [<a href="https://researchcomputing.princeton.edu/education/training/hardware-and-software-requirements-picscie-workshops" target="_blank">click here</a> for help]

+ Please SSH to Adroit in the terminal: `ssh <YourNetID>@adroit.princeton.edu` [click [here](https://researchcomputing.princeton.edu/faq/why-cant-i-login-to-a-clu) for help]. If you have an account on TigerGPU or Traverse then you may use that if the queue is not long.

+ If you are new to Linux then consider using the MyAdroit web portal (VPN required): [https://myadroit.princeton.edu](https://myadroit.princeton.edu)

+ Run the [checkquota](https://researchcomputing.princeton.edu/checkquota) command to make sure that you have free space

+ Go to the [main page](https://github.com/PrincetonUniversity/intro_ml_libs) of this repo

## Where to store your Conda environments

Performing a Conda install of one of the libraries in this repo requires between 2-6 GB. If you plan on installing one or more libraries during the live workshop then you should create a `.condarc` file to direct the installation to either `/scratch/network` or `/scratch/gpfs`:

### Adroit

Create your file as follows:

```
$ cat ~/.condarc
pkgs_dirs:
 - /scratch/network/<YourNetID>/ML_WORKSHOP/conda-pkgs
envs_dirs:
 - /scratch/network/<YourNetID>/ML_WORKSHOP/conda-envs
```

### Della, Perseus, Tiger or Traverse

Create your file as follows:

```
$ cat ~/.condarc
pkgs_dirs:
 - /scratch/gpfs/<YourNetID>/ML_WORKSHOP/conda-pkgs
envs_dirs:
 - /scratch/gpfs/<YourNetID>/ML_WORKSHOP/conda-envs
```

At the end of the workshop you can delete the `.condarc` file and the `ML_WORKSHOP` directory:

```
$ rm -rf ~/.condarc /scratch/network/<YourNetID>/ML_WORKSHOP
```
