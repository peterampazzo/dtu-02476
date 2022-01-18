#!/bin/sh
set -e

gcloud auth activate-service-account 562525590119-compute@developer.gserviceaccount.com --key-file=/app/gcloud-service-key.json --project=dtumlops-338418
gcsfuse dtumlopsdata /app/data #or /root/data???
python -u src/data/make_dataset.py