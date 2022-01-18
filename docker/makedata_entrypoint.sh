#!/bin/sh
set -e

gcloud auth activate-service-account 562525590119-compute@developer.gserviceaccount.com --key-file=gcp_key.json --project=dtumlops-338418
gcsfuse dtumlopsdata /data #or /root/data???
python -u src/data/make_dataset.py