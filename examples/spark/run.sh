#!/bin/bash

function usage {
cat << EOF

    Usage: $0 -[ e ] -c

    -e : Run example job

    -c : Compile with SBT

EOF
    exit 1
}

function remove_hdfs_out {
    echo 'removing' ${OUT_DIR}/${JOB_NAME} 'in HDFS!'
    $HADOOP_HOME/bin/hadoop fs -rm -R ${OUT_DIR}/${JOB_NAME}
}

function spark_runner {
    remove_hdfs_out
    echo 'input:' ${INPUT}
    echo 'output:' ${OUTPUT}
    $SPARK_HOME/bin/spark-submit --master ${MASTER} --deploy-mode cluster \
    --class ${JOB_CLASS} ${JAR_FILE} ${INPUT} ${OUTPUT}
}

# Compile src 
if [[ $* = *-c* ]]; then
    sbt package
    LINES=`find . -name "*.scala" -print | xargs wc -l | grep "total" | awk '{$1=$1};1'`
    echo Project has "$LINES" lines
fi

# Configuration Details

CORE_HDFS="hdfs://olympia:32351"

# MASTER="yarn"
# MASTER="local"
MASTER="spark://olympia.cs.colostate.edu:32365"
OUT_DIR="/out"
JAR_FILE="target/scala-2.11/*.jar"

# Various Jobs to Execute
case "$1" in

-e|--example)
    JOB_NAME="example"
    JOB_CLASS="cs535.spark.ExampleApp"
    INPUT="${CORE_HDFS}/data/example"
    OUTPUT="${CORE_HDFS}${OUT_DIR}/${JOB_NAME}"
    spark_runner
    ;;

*) usage;
    ;;

esac
