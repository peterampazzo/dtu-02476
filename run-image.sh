#!/bin/sh

docker run -d \
  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
  --mount type=bind,src="$(pwd)"/models,dst=/app/models \
  -d train:latest