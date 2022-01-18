#!/bin/sh
set -e

gcloud iam service-account keys create my_key.json --iam-account=562525590119-compute@developer.gserviceaccount.com
gcloud auth activate-service-account 562525590119-compute@developer.gserviceaccount.com --key-file=my_key.json --project=dtumlops-338418
gcsfuse dtumlopsdata /data #or /root/data???
python -u src/data/make_dataset.py