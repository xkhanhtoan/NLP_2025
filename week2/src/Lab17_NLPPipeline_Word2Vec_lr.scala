package com.quangviet.spark
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter, FileWriter}

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.classification.LogisticRegression



object Lab17_NLPPipeline_Word2Vec_lr {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // 1. --- Read Dataset ---
    val dataPath = "D:\\Study\\NLP\\Lab2\\data\\c4-train.00000-of-01024-30K.json"
    val initialDF = spark.read.json(dataPath).limit(1000) 
    println(s"Successfully read ${initialDF.count()} records.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false) 


    val dfWithLabel = initialDF.withColumn("label", 
      when(length($"text") > 500, 1.0).otherwise(0.0)
    )
    println("\nDataFrame with labels:")
    dfWithLabel.select("label", "text").show(5, truncate = 50)

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    // val tokenizer = new RegexTokenizer()
    //   .setInputCol("text")
    //   .setOutputCol("tokens")
    //   .setPattern("\\s+|[.,;!?()\"']") // Fix: Use \\s for regex, and \" for double quote

    
    // Alternative Tokenizer: A simpler, whitespace-based tokenizer.
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")


    // Using Word2Vec for vectorization
    val word2Vec = new Word2Vec()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")
      .setVectorSize(100)  
      .setMinCount(2)       
      .setMaxIter(10)       
      .setSeed(42)          
    
    println("\n=== Using Word2Vec for vectorization ===")
    println(s"Vector size: 100 dimensions")
    println(s"Min word count: 2")


    // Logistic Regression for text classification
    val lr = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.01)    
      .setElasticNetParam(0.0)  /
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
    
    println("\n=== Adding Logistic Regression classifier ===")
    println(s"Max iterations: 20")
    println(s"Regularization: 0.01")

    // 6. --- Assemble the Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, word2Vec, lr))

    // Split data for training and testing
    val Array(trainingData, testData) = dfWithLabel.randomSplit(Array(0.8, 0.2), seed = 42)
    println(s"\nTraining set size: ${trainingData.count()}")
    println(s"Test set size: ${testData.count()}")
    
    // --- Time the main operations ---

    println("\nFitting the NLP pipeline with Word2Vec and Logistic Regression...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(trainingData)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming training data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedTrainDF = pipelineModel.transform(trainingData)
    transformedTrainDF.cache()
    val transformCount = transformedTrainDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Transform test data and evaluate
    println("\nTransforming test data...")
    val transformedTestDF = pipelineModel.transform(testData)
    transformedTestDF.cache()

    // Calculate actual vocabulary size
    val actualVocabSize = transformedTrainDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after preprocessing: $actualVocabSize unique terms.")

    // --- Evaluate Model Performance ---
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    
    val trainAccuracy = evaluator.evaluate(transformedTrainDF)
    val testAccuracy = evaluator.evaluate(transformedTestDF)
    
    println(f"\n=== Model Performance ===")
    println(f"Training Accuracy: ${trainAccuracy * 100}%.2f%%")
    println(f"Test Accuracy: ${testAccuracy * 100}%.2f%%")

    // --- Show Results ---
    println("\nSample of transformed data with predictions:")
    transformedTestDF.select("text", "label", "prediction", "features")
      .show(5, truncate = 50)

    val n_results = 20
    val results = transformedTestDF.select("text", "label", "prediction", "features")
      .take(n_results)

    // 7. --- Write Metrics and Results to Separate Files ---

    // Write metrics to the log folder
    val log_path = "./log/lab17_metrics_word2vec_lr.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new FileWriter(log_path, true))
    try {
      logWriter.println("="*80)
      logWriter.println("=== EXERCISE 3 & 4: Word2Vec + Logistic Regression ===")
      logWriter.println("="*80)
      logWriter.println("\n--- Performance Metrics ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"\n--- Word2Vec Configuration ---")
      logWriter.println(s"Vector size: 100 dimensions")
      logWriter.println(s"Min word count: 2")
      logWriter.println(s"Max iterations: 10")
      logWriter.println(s"\n--- Logistic Regression Configuration ---")
      logWriter.println(s"Max iterations: 20")
      logWriter.println(s"Regularization parameter: 0.01")
      logWriter.println(s"\n--- Model Performance ---")
      logWriter.println(f"Training Accuracy: ${trainAccuracy * 100}%.2f%%")
      logWriter.println(f"Test Accuracy: ${testAccuracy * 100}%.2f%%")
      logWriter.println(s"\n--- Dataset Split ---")
      logWriter.println(s"Training set size: ${trainingData.count()}")
      logWriter.println(s"Test set size: ${testData.count()}")
      logWriter.println(s"\nMetrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Write data results to the results folder
    val result_path = "./results/lab17_pipeline_output_word2vec_lr.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new FileWriter(result_path, true))
    try {
      resultWriter.println("="*80)
      resultWriter.println(s"=== NLP Pipeline Output with Word2Vec + Logistic Regression ===")
      resultWriter.println(s"=== (First $n_results test results) ===")
      resultWriter.println("="*80)
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val label = row.getAs[Double]("label")
        val prediction = row.getAs[Double]("prediction")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"True Label: $label")
        resultWriter.println(s"Predicted Label: $prediction")
        resultWriter.println(s"Correct: ${if (label == prediction) "✓" else "✗"}")
        resultWriter.println(s"Word2Vec Embedding (first 10 dims): ${features.toArray.take(10).mkString(", ")}...")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      
      resultWriter.println("\n" + "="*80)
      resultWriter.println("=== Summary ===")
      resultWriter.println("="*80)
      resultWriter.println(f"Overall Test Accuracy: ${testAccuracy * 100}%.2f%%")
      val correctPredictions = results.count(row => 
        row.getAs[Double]("label") == row.getAs[Double]("prediction")
      )
      resultWriter.println(f"Correct predictions in sample: $correctPredictions / $n_results")
      
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }

    spark.stop()
    println("Spark Session stopped.")
  }
}