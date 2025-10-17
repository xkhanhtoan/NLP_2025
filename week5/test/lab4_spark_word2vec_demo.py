from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec

spark = SparkSession.builder.appName("Lab4_Spark_Word2Vec_Demo").getOrCreate()

DATA_PATH = "C:/Users/Z/Desktop/2025-code/NLP/week5/data/c4-train.00000-of-01024-30K.json"

rdd = spark.sparkContext.textFile(DATA_PATH)
rdd = rdd.filter(lambda line: line.strip() != "" and not line.startswith("#"))
df = rdd.map(lambda line: (line,)).toDF(["text"])

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)

word2Vec = Word2Vec(vectorSize=100, minCount=3, inputCol="words", outputCol="result")
model = word2Vec.fit(wordsData)

result = model.transform(wordsData)
result.select("text", "result").show(3, truncate=False)

for synonym, cosineSim in model.findSynonyms("president", 5).collect():
    print(f"{synonym:10s} -> {cosineSim:.4f}")

spark.stop()
