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

docker run -d \
  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
  --mount type=bind,src="$(pwd)"/models,dst=/app/models \
  -d $image