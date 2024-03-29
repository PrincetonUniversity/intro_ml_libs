# Spark

[Spark](https://spark.apache.org/) is a big data processing engine with specialized components for machine learning, SQL, graphs, streaming and more. Its greatest utility is that the parallelization and processing of the data is automatically handled. The are four different Spark API languages: Scala, Python, Java and R.

- The basic idea is to store the data in a DataFrame which is distributed over many nodes. The parallelization is handled automatically. The underlying data structure is the resilent distributed datasets (RDD). Think of an RDD of a list of objects.

- Lazy evaluation is used. Operations in the Spark script which transform an RDD are translated to a node in a computation graph instead of being immediately evaluated. Actions cause the graph to be evaluated. Intermediate results can be cached in memory and/or disk.

Spark 3 is available on the Princeton HPC clusters. See the [Python API](https://spark.apache.org/docs/3.2.0/api/python/) and the Research Computing [knowledge base page](https://researchcomputing.princeton.edu/support/knowledge-base/spark) for Spark.

## A Simple DataFrame

The session below illustrates how to create a simple DataFrame in the PySpark shell during an interactve session:

```bash
$ ssh <YourNetID>@adroit.princeton.edu  # or another cluster
$ salloc --nodes=1 --ntasks=1 --time=10
$ module load anaconda3/2022.10 spark/hadoop3.2/3.2.0  # do not use a newer anaconda3 module
$ spark-start
$ pyspark

>>> myRDD = sc.parallelize([('Denver', 5280), ('Albuquerque', 5312), ('Mexico City', 7382)])
>>> df = sqlContext.createDataFrame(myRDD, ['City', 'Elevation'])
>>> df.show()
+-----------+---------+
|       City|Elevation|
+-----------+---------+
|     Denver|     5280|
|Albuquerque|     5312|
|Mexico City|     7382|
+-----------+---------+
>>> df = df.filter(df["Elevation"] < 6000)
>>> df.show()
+-----------+---------+
|       City|Elevation|
+-----------+---------+
|     Denver|     5280|
|Albuquerque|     5312|
+-----------+---------+
>>> exit()
$ exit
```

## Hello World with the Slurm Job Scheduler

If you are new to Spark then start by running this simple example:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/network/<YourNetID>  # /scratch/gpfs/ on other clusters
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop.git
$ cd hpc_beginning_workshop/spark_big_data
$ wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py
$ sbatch job.slurm  # edit email address
```

## Machine Learning

#### Python Example

Spark MLlib is the machine learning component of Spark. See documentaion for the Python API of [Spark ML 3.2.0](https://spark.apache.org/docs/3.2.0/ml-guide.html). This is the dataframe-based API known as `spark.ml`.

The Spark 3.2 machine learning examples are here:

```bash
# ssh tiger, della, stellar, adroit
$ cd /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main
$ ls
java  python  r  resources  scala  scripts
$ cd /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/python/ml
$ ls
aft_survival_regression.py                   max_abs_scaler_example.py
als_example.py                               min_hash_lsh_example.py
binarizer_example.py                         min_max_scaler_example.py
bisecting_k_means_example.py                 multiclass_logistic_regression_with_elastic_net.py
bucketed_random_projection_lsh_example.py    multilayer_perceptron_classification.py
bucketizer_example.py                        naive_bayes_example.py
chisq_selector_example.py                    n_gram_example.py
chi_square_test_example.py                   normalizer_example.py
correlation_example.py                       onehot_encoder_example.py
count_vectorizer_example.py                  one_vs_rest_example.py
cross_validator.py                           pca_example.py
dataframe_example.py                         pipeline_example.py
dct_example.py                               polynomial_expansion_example.py
decision_tree_classification_example.py      power_iteration_clustering_example.py
decision_tree_regression_example.py          prefixspan_example.py
elementwise_product_example.py               quantile_discretizer_example.py
estimator_transformer_param_example.py       random_forest_classifier_example.py
feature_hasher_example.py                    random_forest_regressor_example.py
fm_classifier_example.py                     rformula_example.py
fm_regressor_example.py                      robust_scaler_example.py
fpgrowth_example.py                          sql_transformer.py
gaussian_mixture_example.py                  standard_scaler_example.py
generalized_linear_regression_example.py     stopwords_remover_example.py
gradient_boosted_tree_classifier_example.py  string_indexer_example.py
gradient_boosted_tree_regressor_example.py   summarizer_example.py
imputer_example.py                           tf_idf_example.py
index_to_string_example.py                   tokenizer_example.py
interaction_example.py                       train_validation_split.py
isotonic_regression_example.py               univariate_feature_selector_example.py
kmeans_example.py                            variance_threshold_selector_example.py
lda_example.py                               vector_assembler_example.py
linear_regression_with_elastic_net.py        vector_indexer_example.py
linearsvc.py                                 vector_size_hint_example.py
logistic_regression_summary_example.py       vector_slicer_example.py
logistic_regression_with_elastic_net.py      word2vec_example.py
```

Below is `random_forest_classifier_example.py`:

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()

    # Load and parse the data file, converting it to a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    spark.stop()
```

Run the commands below:

```bash
$ git clone https://github.com/PrincetonUniversity/intro_ml_libs
$ cd intro_ml_libs/spark
$ cp /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/python/ml/random_forest_classifier_example.py .
$ cp /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/data/mllib/sample_libsvm_data.txt .
# next line corrects the path to sample_libsvm_data.txt
$ sed -i 's$data/mllib/$$g' random_forest_classifier_example.py
```

Now submit the job:

```bash
$ sbatch job.submit
```

You can see the updated examples on [GitHub](https://github.com/apache/spark/tree/master/examples/src/main).

```bash
#!/bin/bash
#SBATCH --job-name=spark-ml      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2022.10
module load spark/hadoop3.2/3.2.0

spark-start
spark-submit --total-executor-cores 6 --executor-memory 4G random_forest_classifier_example.py
```

The output of the run is:

```bash
$ cat slurm-*.out | grep -v INFO
...
  total: 1.503044646
  findBestSplits: 1.48605558
  chooseSplits: 1.478890107
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[100,101,102...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[125,126,127...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows

Test Error = 0.04
RandomForestClassificationModel: uid=RandomForestClassifier_a139e7bca298, numTrees=10, numClasses=2, numFeatures=692
...
```

#### R Example

See this directory for examples: `/usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/r/ml`

Here is the random forest example:

```
library(SparkR)

# Initialize SparkSession
sparkR.session(appName = "SparkR-ML-randomForest-example")

# Load training data
df <- read.df("data/mllib/sample_libsvm_data.txt", source = "libsvm")
training <- df
test <- df

# Fit a random forest classification model with spark.randomForest
model <- spark.randomForest(training, label ~ features, "classification", numTrees = 10)

# Model summary
summary(model)

# Prediction
predictions <- predict(model, test)
head(predictions)

# Random forest regression model

# Load training data
df <- read.df("data/mllib/sample_linear_regression_data.txt", source = "libsvm")
training <- df
test <- df

# Fit a random forest regression model with spark.randomForest
model <- spark.randomForest(training, label ~ features, "regression", numTrees = 10)

# Model summary
summary(model)

# Prediction
predictions <- predict(model, test)
head(predictions)

sparkR.session.stop()
```

Run the commands below to prepare the input files:

```bash
$ mkdir R_example && cd R_example
$ cp /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/examples/src/main/r/ml/randomForest.R .
$ cp /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/data/mllib/sample_libsvm_data.txt .
$ cp /usr/licensed/spark/spark-3.2.0-bin-hadoop3.2/data/mllib/sample_linear_regression_data.txt .
# next line corrects the path to data files
$ sed -i 's$data/mllib/$$g' randomForest.R
```

Below is an appropriate Slurm script (job.slurm):

```bash
#!/bin/bash
#SBATCH --job-name=spark-ml      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load spark/hadoop3.2/3.2.0

spark-start
spark-submit --total-executor-cores 2 --executor-memory 4G randomForest.R
```

Submit the job with:

```
$ sbatch job.slurm
```


## Spark at Princeton

[Spark tutorial](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-tut)  
[Tuning Spark applications](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-memory)  
[Spark application submission via Slurm](https://researchcomputing.princeton.edu/faq/spark-via-slurm)

## More links

[Sparkling Water](https://www.h2o.ai/products/h2o-sparkling-water/)  
[Deep Learning Pipelines](https://github.com/databricks/spark-deep-learning)
