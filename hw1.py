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
# Initial params
limit = 50  # Use -1 for no limit
data = "barcelona"  # Use "barcelona" or "titles"
nearest = 5

# TODO what is the hyperparameters?
features = "tfidf"  # Use "tfidf" or "word2vec"
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
    indices = [Vectors.dense(df["id"][np.delete(ind, np.where(ind == i))].values) for i, ind in enumerate(indices)]
    df[output_col] = indices

    return spark.createDataFrame(df)



@timeit
def lsh_prediction(ds, k=nearest, input_col="features", output_col="ann_neighbors", num_hash_tables=numHashTables):
    model = MinHashLSH(inputCol=input_col, outputCol=output_col, numHashTables=num_hash_tables)
    model = model.fit(ds)

    # ds = model.transform(ds) # TODO what is this for?

    anns = []
    for i in tqdm(ds.collect()):
        id_ = i["id"]
        key = i[input_col]

        anns.append(model.approxNearestNeighbors(ds, key, k+1).filter(col("id") != id_).select("id").collect())

    anns = [Vectors.dense([i["id"] for i in ann]) for ann in anns]
    df = ds.toPandas()
    df[output_col] = anns
    ds = spark.createDataFrame(df)


    return ds

def eval(ds):
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
    spark = SparkSession(sc)

    print(spark)

    print("Reading data for " + data + " with limit " + str(limit) + " and features " + features)
    data = read_data(spark)
    data = limit_data(data)

    data = get_features(data)

    print("Calculating gt neighbors nearest neighbors, could take a while...")
    data = compute_gt(data)
    print(data.select("gt_neighbors").show(5))

    print("Calculating lsh neighbors nearest neighbors, could take a while...")
    data = lsh_prediction(data)

    print("Accuracy: " + str(eval(data)))
