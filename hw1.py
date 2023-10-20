import pyspark
from tqdm import tqdm
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, MinHashLSH
from pyspark.ml.feature import Word2Vec
import findspark
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pyspark.sql.functions import col

findspark.init()
stop_list = ['the', 'a', 'an', 'another', "for", "an", "nor", "but", "or", "yet", "so",
             "in", "under", "towards", "before"]

# TODO move to args?
# TODO set all seeds
# Initial params
limit = 50  # Use -1 for no limit
data = "barcelona"  # Use "barcelona" or "titles"
nearest = 5

# TODO what is the hyperparameters?
type_features = "tfidf"  # Use "tfidf" or "word_to_vec"
use_stopwords = True  # Use True or False
use_custom_stopwords = False  # Use True or False
latent_features = 20  # Dimension of features
numHashTables = 3


def timeit(func):
    def timed(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Time elapsed for " + func.__name__ + ": " + str(end - start))
        return result

    return timed


def read_data(spark):
    if data == "barcelona":
        df = spark.read.csv("listings_barcelona.csv", header=True, multiLine=True)
    elif data == "titles":
        raise NotImplementedError  # TODO implement loading of titles from wiki
    else:
        raise ValueError("Invalid data")
    return df


def limit_data(df):
    if limit > 0:
        df = df.limit(limit)
    return df


def get_features(df, input_col="name", output_col="features", type_features=type_features):
    tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
    df = tokenizer.transform(df)

    if use_stopwords:
        if use_custom_stopwords:
            remover = StopWordsRemover(stopWords=stop_list, inputCol="words", outputCol="clean_tokens")
        else:
            remover = StopWordsRemover(inputCol="words", outputCol="clean_tokens")
        df = remover.transform(df)
        df = df.drop("words")
        df = df.withColumnRenamed("clean_tokens", "words")

    if type_features == "tfidf":
        hashing = HashingTF(inputCol="words", outputCol="hash", numFeatures=latent_features)
        df = hashing.transform(df)

        idf = IDF(inputCol="hash", outputCol=output_col)
        model = idf.fit(df)
        df = model.transform(df)
    elif type_features == "word_to_vec":
        word_vec = Word2Vec(vectorSize=latent_features, minCount=0, inputCol="words", outputCol=output_col)
        model = word_vec.fit(df)
        df = model.transform(df)
    else:
        raise ValueError("Invalid feature " + type_features)

    return df


@timeit
def compute_gt(ds, spark, k=nearest, input_col="features", output_col="gt_neighbors"):
    # TODO make it fully on spark
    df = ds.toPandas()
    features = df[input_col].tolist()
    model = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(features)
    distances, indices = model.kneighbors(features)
    # remove self from neighbors TODO something better and faster?
    indices = [Vectors.dense(df["id"][np.delete(ind, np.where(ind == i))].values) for i, ind in enumerate(indices)]
    df[output_col] = indices

    return spark.createDataFrame(df)


@timeit
def lsh_prediction(ds, spark, k=nearest, input_col="features",
                   output_col="ann_neighbors", num_hash_tables=numHashTables):
    model = MinHashLSH(inputCol=input_col, outputCol=output_col, numHashTables=num_hash_tables)
    model = model.fit(ds)

    # TODO There should be something better than this
    pred = []
    for i in tqdm(ds.collect()):
        id_ = i["id"]
        key = i[input_col]

        pred.append(model.approxNearestNeighbors(ds, key, k + 1).filter(col("id") != id_).select("id").collect())

    pred = [Vectors.dense([i["id"] for i in ann]) for ann in pred]
    df = ds.toPandas()
    df[output_col] = pred
    ds = spark.createDataFrame(df)

    return ds


def evaluation(ds):
    # TODO There should be something better than this
    acc = 0
    for i in tqdm(ds.collect()):
        gt = i["gt_neighbors"]
        ann = i["ann_neighbors"]
        gt.sort(), ann.sort()
        acc += len(set(gt).intersection(set(ann)))
    acc /= len(ds.collect()) * nearest

    return acc


if __name__ == '__main__':
    sc = pyspark.SparkContext('local[*]')
    sc = SparkSession(sc)

    print(sc)

    print("Reading data for " + data + " with limit " + str(limit) + " and features " + type_features)
    data = read_data(sc)
    data = limit_data(data)

    data = get_features(data)

    print("Calculating gt neighbors nearest neighbors, could take a while...")
    data = compute_gt(data, sc)
    print(data.select("gt_neighbors").show(5))

    print("Calculating lsh neighbors nearest neighbors, could take a while...")
    data = lsh_prediction(data, sc)

    # TODO val split?
    print("Accuracy: " + str(evaluation(data)))
