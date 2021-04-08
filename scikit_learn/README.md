# Scikit-learn

[Scikit-learn](https://scikit-learn.org/stable/) is a Python library for machine learning featuring:

+ Conventional models (not a deep learning library)
+ Takes full advantage of the flexibility of Python
+ Interoperable with Pandas
+ Single-node only with multithreading, some GPU support (e.g., XGBoost)
+ Excellent [documentation](https://scikit-learn.org/stable/user_guide.html) (good way to prepare for ML engineer and data scientist interviews)

For an introduction to machine learning and Scikit-learn see this GitHub [repo and book](https://github.com/ageron/handson-ml2) by Aurelien Geron.

## Installation

### Anaconda

Scikit-learn is pre-installed as part of the Anaconda Python disribution:

```python
$ module load anaconda3/2020.11
(base) $ python
>>> import sklearn
>>> print(sklearn.__version__)
'0.23.2'
```

If you need additional packages that are not found in the Anaconda distribution then make your own Conda environment:

```bash
$ module load anaconda3/2020.11
$ conda create --name sklearn-env --channel <some-channel> scikit-learn pandas matplotlib <another-package>
```

See [this page](https://researchcomputing.princeton.edu/python) for more on creating Conda environments for Python packages and writing Slurm scripts.

### Intel

Intel provides their own distribution of Python as well as acceleration libraries for Scikit-learn such as [DAAL](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html). You may consider creating your Scikit-learn environment using packages from the `intel` channel:

```bash
$ module load anaconda3/2020.11
$ conda create --name sklearn-env --channel intel scikit-learn pandas matplotlib
```

## Multithreading

Scikit-learn depends on the `intel-openmp` package which enabling multithreading. This allows the software to use multiple CPU-cores for hyperparameter tuning, cross validation and other embarrassingly parallel operations. If you are calling a routine that takes the `n_jobs` parameter then set this to `n_jobs=-1` to take advantage of all the CPU-cores in your Slurm allocation.

Below is an appropriate Slurm script for a Scikit-learn job that takes advantage of multithreading:

```bash
#!/bin/bash
#SBATCH --job-name=sklearn       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=<YourNetID>@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2020.11
conda activate sklearn-env

python myscript.py
```

## Example Job

Sentiment analysis on [movie reviews](http://ai.stanford.edu/~amaas/data/sentiment/). Here are a few samples:

```
                                              review  sentiment
0  Zentropa has much in common with The Third Man...          1
1  Zentropa is the most original movie I've seen ...          1
2  Lars Von Trier is never backward in trying out...          1
3  *Contains spoilers due to me having to describ...          1
4  That was the first thing that sprang to mind a...          1
```

A code to train a logistic regression model using bag-of-words and TF-IDF is below:

```python
import os
import numpy as np
import pandas as pd

labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('train', 'test'):
  for l in ('pos', 'neg'):
    path = 'aclImdb/' + s + '/' + l
    for file in os.listdir(path):
      with open(os.path.join(path, file), 'r') as infile:
        txt = infile.read()
      df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']

import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer

# use the partial module to remove duplicate code from these two methods
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return " ".join(words)

def review_to_words_porter(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    porter = PorterStemmer()
    return " ".join(porter.stem(word) for word in words)

split = int(0.6 * df.shape[0])
X_train = df.iloc[:split, 0].values
y_train = df.iloc[:split, 1].values
X_test = df.iloc[split:, 0].values
y_test = df.iloc[split:, 1].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
stops = stopwords.words("english")

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, max_features=7500)
param_grid = [{'vect__ngram_range': [(1, 3)], 'vect__stop_words': [stops, None],
               'vect__tokenizer': [review_to_words, review_to_words_porter],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 3)], 'vect__stop_words': [stops, None],
               'vect__tokenizer': [review_to_words, review_to_words_porter],
               'clf__C': [1.0, 10.0, 100.0],
               'vect__use_idf': [False], 'vect__norm': [None]}]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(max_iter=250))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print(gs_lr_tfidf.best_params_)

clf = gs_lr_tfidf.best_estimator_
print('Accuracy (train): %.1f percent' % (100 * clf.score(X_train, y_train)))
print('Accuracy (test): %.1f percent' % (100 * clf.score(X_test, y_test)))
```

The output is:

```
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.

Accuracy (train): 88.9 percent
Accuracy (test): 84.1 percent
```

Below is the effect of CPU-cores on the run time:

| cpus-per-task | time |
|:----:|:-----:|
| 4  | 56:19 |
| 16 | 20:22 |


## Another Example

Take a look at an end-to-end ML project [here](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb) for predicting housing prices in California.


## Gradient Boosting Models and GPUs

[XGBoost](https://xgboost.readthedocs.io/en/latest/) and the Scikit-learn [interface](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

[LightGBM](https://github.com/microsoft/LightGBM/tree/master/python-package)

See the [Intel implementation](https://software.intel.com/content/www/us/en/develop/tools/data-analytics-acceleration-library.html) of XGB
