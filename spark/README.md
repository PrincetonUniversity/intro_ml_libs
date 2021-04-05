# Spark

[Spark](https://spark.apache.org/) is a big data processing engine with specialized components for machine learning, SQL, graphs, streaming and more. Its greatest utility is that the parallelization and processing of the data is automatically handled. The are four different Spark API languages: Scala, Python, Java and R.

- The basic idea is to store the data in a DataFrame which is distributed over many nodes. The parallelization is handled automatically. The underlying data structure is the resilent distributed datasets (RDD). Think of an RDD of a list of objects.

- Lazy evaluation is used. Operations in the Spark script which transform an RDD are translated to a node in a computation graph instead of being immediately evaluated. Actions cause the graph to be evaluated. Intermediate results can be cached in memory and/or disk.

Spark 2.4 is available on the Princeton HPC clusters. See the [Python API](https://spark.apache.org/docs/2.4.6/api/python/index.html). Spark will not work on our clusters with Anaconda modules newer than anaconda3/2019.10.

## A Simple DataFrame

The session below illustrates how to create a simple DataFrame in the PySpark shell during an interactve session:

```bash
$ ssh <YourNetID>@adroit.princeton.edu  # or another cluster
$ salloc --nodes=1 --ntasks=1 --time=10
$ module load anaconda3/2019.10 spark/hadoop2.7/2.4.6
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
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop
$ cd hpc_beginning_workshop/RC_example_jobs/spark_big_data
$ wget https://raw.githubusercontent.com/apache/spark/master/examples/src/main/python/pi.py
$ sbatch job.slurm  # edit email address
```

## Machine Learning

Spark ML is the machine learning component of Spark. The previous library was called mllib.

The documentaion for the Python API with [Spark ML 2.4](https://spark.apache.org/docs/2.4.6/api/python/pyspark.ml.html) is here.

The Spark 2.4 machine learning examples are here:

```bash
# ssh tiger, della, perseus, adroit
$ cd /usr/licensed/spark/spark-2.4.6-bin-hadoop2.7/examples/src/main
$ ls
java  python  r  resources  scala
$ cd /usr/licensed/spark/spark-2.4.6-bin-hadoop2.7/examples/src/main/python/ml
$ ls
aft_survival_regression.py                   logistic_regression_with_elastic_net.py
als_example.py                               max_abs_scaler_example.py
binarizer_example.py                         min_hash_lsh_example.py
bisecting_k_means_example.py                 min_max_scaler_example.py
bucketed_random_projection_lsh_example.py    multiclass_logistic_regression_with_elastic_net.py
bucketizer_example.py                        multilayer_perceptron_classification.py
chisq_selector_example.py                    naive_bayes_example.py
chi_square_test_example.py                   n_gram_example.py
correlation_example.py                       normalizer_example.py
count_vectorizer_example.py                  onehot_encoder_example.py
cross_validator.py                           one_vs_rest_example.py
dataframe_example.py                         pca_example.py
dct_example.py                               pipeline_example.py
decision_tree_classification_example.py      polynomial_expansion_example.py
decision_tree_regression_example.py          quantile_discretizer_example.py
elementwise_product_example.py               random_forest_classifier_example.py
estimator_transformer_param_example.py       random_forest_regressor_example.py
fpgrowth_example.py                          rformula_example.py
gaussian_mixture_example.py                  sql_transformer.py
generalized_linear_regression_example.py     standard_scaler_example.py
gradient_boosted_tree_classifier_example.py  stopwords_remover_example.py
gradient_boosted_tree_regressor_example.py   string_indexer_example.py
imputer_example.py                           tf_idf_example.py
index_to_string_example.py                   tokenizer_example.py
isotonic_regression_example.py               train_validation_split.py
kmeans_example.py                            vector_assembler_example.py
lda_example.py                               vector_indexer_example.py
linear_regression_with_elastic_net.py        vector_slicer_example.py
linearsvc.py                                 word2vec_example.py
logistic_regression_summary_example.py
```

Below is `random_forest_classifier_example.py`:

```python
from __future__ import print_function

# $example on$
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("RandomForestClassifierExample")\
        .getOrCreate()

    # $example on$
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
    # $example off$

    spark.stop()
```

Run the commands below:

```bash
$ git clone https://github.com/PrincetonUniversity/intro_ml_libs
$ cd intro_ml_libs/spark
$ cp /usr/licensed/spark/spark-2.4.6-bin-hadoop2.7/examples/src/main/python/ml/random_forest_classifier_example.py .
$ cp /usr/licensed/spark/spark-2.4.6-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt .
```

Use a text editor to replace line 39 of `random_forest_classifier_example.py` with this:

```bash
    data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
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
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2019.10
module load spark/hadoop2.7/2.4.6

spark-start
spark-submit --total-executor-cores 6 --executor-memory 4G random_forest_classifier_example.py
```

The output of the run is:

```bash
$ cat slurm-*.out | grep -v INFO
...
  total: 4.680999166
  findSplits: 2.825805252
  findBestSplits: 1.183317259
  chooseSplits: 1.179836058
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[98,99,100,1...|
|           0.0|  0.0|(692,[100,101,102...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows

Test Error = 0
...
```

### Spark at Princeton

[Spark tutorial](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-tut)  
[Tuning Spark applications](https://researchcomputing.princeton.edu/computational-hardware/hadoop/spark-memory)  
[Spark application submission via Slurm](https://researchcomputing.princeton.edu/faq/spark-via-slurm)

### More links

[Sparkling Water](https://www.h2o.ai/products/h2o-sparkling-water/)  
[Deep Learning Pipelines](https://github.com/databricks/spark-deep-learning)
