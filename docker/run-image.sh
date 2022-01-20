#!/bin/sh

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi

if [ -z "$2" ]
  then
    tag="latest"
  else
    tag=$2
fi

image="${1}:${tag}"
echo "Running: $image";

# --mount type=bind,src="$(pwd)"/data,dst=/app/data \
# --mount type=bind,src="$(pwd)"/models,dst=/app/models \

docker run -d \
  --privileged \
  --env-file .env \
  --mount type=bind,src="$(pwd)"/cred.json,dst=/app/gcloud-service-key.json \
  -d $image