package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.evaluation.ClusteringEvaluator

import org.apache.spark.sql.SparkSession
import scala.util.control.Breaks._

object kmeans {
  def main(args: Array[String]) {
	println("Direct k-means clustering in Scala/Spark");

	val spark = SparkSession.builder.appName("kmeans").getOrCreate()

	// Load and parse the data
	val path = "/project/ch1-7-12-scaled.csv"
	println("Reading " + path + "...")
	val data = spark.read.textFile(path).rdd
	val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

	// Cluster the data into classes using KMeans
	val numClusters = 4
	val numIterations = 2000
	val clusters = KMeans.train(parsedData, numClusters, numIterations)

	// Evaluate clustering by computing Within Set Sum of Squared Errors
	val WSSSE = clusters.computeCost(parsedData)
	println(s"Within Set Sum of Squared Errors (k=$numClusters)= $WSSSE")

//	val predictions = clusters.predict(parsedData);
//	val evaluator = new ClusteringEvaluator()
//	val silhouette = evaluator.evaluate(predictions)
//	println(s"Silhouette with squared euclidean distance = $silhouette")

	for (i <- 0 to 3) {
		println(s"Cluster $i center: " + clusters.clusterCenters(i));
	}

	// Save model
//	println("Writing model to disk")
//	clusters.save(spark.sparkContext, "/project/KMeansModel")
//	val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")

	spark.stop()


	println("Program complete.");
  }
}
