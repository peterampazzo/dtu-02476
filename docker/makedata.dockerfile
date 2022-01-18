# docker build -f makedata.dockerfile . -t data:latest  
#docker run -d \
#  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
#  --mount type=bind,src="$(pwd)"/models,dst=/app/models \
#  -d data:latest

# Base image
FROM python:3.7-slim

ARG GCP_KEY

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install system packages.
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y gnupg

# Install gcsfuse.
RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y gcsfuse

# Install gcloud.
RUN apt-get install -y apt-transport-https
RUN apt-get install -y ca-certificates
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN apt-get install -y google-cloud-sdk

WORKDIR /app
RUN printf '%s' "$GCP_KEY" > /app/gcloud-service-key.json

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY config.yaml config.yaml
COPY docker/makedata_entrypoint.sh makedata_entrypoint.sh

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e .

ENTRYPOINT ["bash", "makedata_entrypoint.sh"]

# CMD ["sh", "-c", "tail -f /dev/null"]