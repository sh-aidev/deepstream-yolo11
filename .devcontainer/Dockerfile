# ARG BASE_DEEPSTREAM_DEV_CONTAINER=nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

FROM nvcr.io/nvidia/deepstream:7.0-gc-triton-devel AS build

RUN apt-get update && \
    apt-get install -y build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py

# # Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt