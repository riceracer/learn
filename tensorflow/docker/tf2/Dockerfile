FROM tensorflow/tensorflow:2.0.0b0-gpu-py3

#RUN apt-get update
# This is set so tzdata doesn't ask where you live during a docker build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y tzdata python3-tk

WORKDIR /dependencies
COPY requirements.txt /dependencies/requirements.txt

# pip upgrade in dockerfile can cause errors. change this?
RUN pip install --upgrade pip
RUN pip install pip-tools
RUN pip install -r /dependencies/requirements.txt

WORKDIR /app
