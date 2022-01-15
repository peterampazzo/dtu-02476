# docker build -f makedata.dockerfile . -t data:latest  
#docker run -d \
#  --mount type=bind,src="$(pwd)"/data,dst=/app/data \
#  --mount type=bind,src="$(pwd)"/models,dst=/app/models \
#  -d data:latest

# Base image
FROM python:3.7-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
#COPY config.yaml config.yaml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e .

ENTRYPOINT ["python", "-u", "src/data/make_dataset.py"]

# CMD ["sh", "-c", "tail -f /dev/null"]