package cs535.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame

/**
 *
 */
object ExampleApp {

  val DEBUG = false;

  /**
   * Driver method 
   *
   * args inputDir, outputDir
   *
   */
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("Experiment - Default Application").getOrCreate()

    val inputDir = args(0)
    val outputDir = args(1)

    val df = spark.read
          .format("csv")
          .option("header", "true") //first line in file has headers
          .option("mode", "DROPMALFORMED")
          .load(inputDir + "/airtravel.csv")

    saveToHDFS(df, outputDir)   
  }

 /**
   * Save results to HDFS.
   *
   */
  def saveToHDFS(df: DataFrame, outputDir: String) = {
    
    if (DEBUG) {
      println("INFO: finished something cool! ---")
    }
    df.coalesce(1).write.csv(outputDir)
  }

}
