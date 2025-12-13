# test/lab5_spark_sentiment_analysis.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datasets import load_dataset


def main():
    spark = SparkSession.builder \
        .appName("SparkSentimentAnalysis") \
        .getOrCreate()

    # LOAD HUGGINGFACE DATASET (NO CSV)
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    df_pd = ds["train"].to_pandas()

    # 0=neg, 1=neutral, 2=pos → drop neutral
    df_pd = df_pd[df_pd["label"] != 1]
    df_pd["sentiment"] = df_pd["label"].map({0: -1, 2: 1})
    df_pd = df_pd[["text", "sentiment"]]

    df = spark.createDataFrame(df_pd)
    df = df.withColumn("label", (col("sentiment") + 1) / 2)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=10000
    )
    idf = IDF(inputCol="raw_features", outputCol="features")

    lr = LogisticRegression(
        maxIter=10,
        regParam=0.001,
        featuresCol="features",
        labelCol="label"
    )

    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords,
        hashing_tf,
        idf,
        lr
    ])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    acc = evaluator.evaluate(predictions)
    print(f"Spark Sentiment Accuracy: {acc:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
