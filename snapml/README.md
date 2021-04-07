# Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/) is a GPU-enabled machine learning library:

+ Developed by IBM
+ Conventional models (GLM and trees)
+ `sklearn` interface
+ GPU acceleration, distributed training and supports sparse data structures
+ `snapml-spark` offers distributed training and integrates with PySpark (linear regression, logistic regression, linear support vector classifier)

See the [documentation](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/index.html) (v1.6.0) and [tutorials](https://ibmsoe.github.io/snap-ml-doc/v1.6.0/tutorials.html).

## Installation

### Traverse

`pai4sk` is an interface that provides the full functionality of sklearn (for what it supports):

```
$ module load anaconda3/2020.11
$ CHNL="https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda"
$ conda create --name pai4sk-env --channel ${CHNL} pai4sk scikit-learn
```

One can also install only `snapml`:

```
$ module load anaconda3/2020.11
$ CHNL="https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access"
$ conda create --name snap-env --channel ${CHNL} snapml scikit-learn
```

### TigerGPU, Adroit

Only the Spark interface is available for the `x86_64` architecture:

```
$ module load anaconda3/2020.11
$ CHNL="https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda"
$ conda create --name snap-env --channel ${CHML} snapml-spark
```

## Example Job

The following job was ran on Traverse:

```python
from pai4sk import RandomForestClassifier as SnapForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

rf = SnapForest(n_estimators=50, n_jobs=4, max_depth=8, use_histograms=True, use_gpu=True, gpu_ids=[0])
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy = {100 * acc:.1f}%")
```

The output of the code should show high accuracy:

```
Accuracy = 95.6%
```

Below is the corresponding Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node

module purge
module load anaconda3/2020.11
conda activate pai4sk-env

python myscript.py
```

## Notes

When trying to do the `iris` dataset instead of `load_breast_cancer`:

```
ValueError: Multiclass classification not supported for decision tree classifiers.
```
