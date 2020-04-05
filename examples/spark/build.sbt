name := "example-app"

version := "1.0"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "2.4.4",
    "org.apache.spark" %% "spark-sql" % "2.4.4",
    "org.apache.spark" %% "spark-mllib" % "2.4.4"
)
