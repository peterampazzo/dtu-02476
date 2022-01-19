#!/bin/sh
set -e

gcloud auth activate-service-account ${GC_USER} --key-file ${GC_KEY} --project ${GC_PROJECT}
mkdir /bucket/
gcsfuse --key-file ${GC_KEY} --implicit-dirs dtumlopsdata /bucket/ #or /root/data???
python -u src/data/make_dataset.py ${INPUT_FOLDER} ${OUTPUT_FOLDER} ${TEST_SIZE}