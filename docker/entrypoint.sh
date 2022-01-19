#!/bin/sh
set -e

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi

gcloud auth activate-service-account ${GC_USER} --key-file ${GC_KEY} --project ${GC_PROJECT}
mkdir /bucket/
gcsfuse --key-file ${GC_KEY} --implicit-dirs dtumlopsdata /bucket/ #or /root/data???


if [ $1 = "make_data" ]
then
    python -u src/data/make_dataset.py ${RAW_DATA} ${PROCESSED_DATA} ${TEST_SIZE}
fi

if [ $1 = "train" ]
then
    python -u src/models/train_model.py ${PROCESSED_DATA} ${MODELS} ${PROFILE}
fi

if [ $1 = "predict" ]
then
    python -u src/models/predict_model.py 
fi
