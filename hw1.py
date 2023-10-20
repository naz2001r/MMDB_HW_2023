import pyspark
from pyspark import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.feature import Word2Vec
import findspark
from sklearn.neighbors import NearestNeighbors
import numpy as np

findspark.init()
stop_list = ['the', 'a', 'an', 'another', "for", "an", "nor", "but", "or", "yet", "so",
             "in", "under", "towards", "before"]

# TODO move to args?
# Initial params
limit = 500  # Use -1 for no limit
data = "barcelona"  # Use "barcelona" or "titles"
nearest = 5

# TODO Is the next hyperparameters?
features = "tfidf"  # Use "tfidf" or "word2vec"
use_stopwords = True  # Use True or False
use_custom_stopwords = False  # Use True or False
latent_features = 20  # Dimension of features



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


def get_features(df, input_col="name", output_col="features"):
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

    if features == "tfidf":
        hashing = HashingTF(inputCol="words", outputCol="hash", numFeatures=latent_features)
        df = hashing.transform(df)

        idf = IDF(inputCol="hash", outputCol=output_col)
        model = idf.fit(df)
        df = model.transform(df)
    elif features == "word2vec":
        word_vec = Word2Vec(vectorSize=latent_features, minCount=0, inputCol="words", outputCol=output_col)
        model = word_vec.fit(df)
        df = model.transform(df)
    else:
        raise ValueError("Invalid feature " + features)

    return df


@timeit
def compute_gt(ds, k=nearest, input_col="features", output_col="gt_neighbors"):
    # TODO make it fully on spark
    df = ds.toPandas()
    features = df[input_col].tolist()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(features)
    # remove self from neighbors TODO something better and faster?
    indices = [Vectors.dense(np.delete(ind, np.where(ind == i))) for i, ind in enumerate(indices)]
    df[output_col] = indices

    return spark.createDataFrame(df)


if __name__ == '__main__':
    sc = pyspark.SparkContext('local[*]')
    spark = SparkSession(sc)

    print(spark)

    print("Reading data for " + data + " with limit " + str(limit) + " and features " + features)
    data = read_data(spark)
    data = limit_data(data)

    data = get_features(data)

    print("Calculating gt neighbors nearest neighbors, could take a while...")
    data = compute_gt(data)
    print(data.select("gt_neighbors").show(5))
