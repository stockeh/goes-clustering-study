package org.apache.spark.examples.mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.sql.SparkSession
import scala.util.control.Breaks._

object kmeans {
  def main(args: Array[String]) {
	println("Direct k-means clustering in Scala/Spark");

	val conf = new SparkConf().setAppName("KMeans")
	val sc = new SparkContext(conf)

	// Load and parse the data
	val data = sc.textFile("sample_data.txt")
	val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

	// Cluster the data into two classes using KMeans
	val numClusters = 2
	val numIterations = 20
	val clusters = KMeans.train(parsedData, numClusters, numIterations)

	// Evaluate clustering by computing Within Set Sum of Squared Errors
	val WSSSE = clusters.computeCost(parsedData)
	println(s"Within Set Sum of Squared Errors = $WSSSE")

	// Save and load model
//	clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
//	val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")

	sc.stop()


	println("Program complete.");
  }
}
