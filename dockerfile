FROM continuumio/miniconda3
RUN conda install python=3.9

# Install base utilities
RUN apt-get update && \
    apt-get install -y \
    wget \
    sudo \
    htop \
    make \
    vim \
    tmux \
    openssh-server \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libgoogle-perftools-dev \
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt
COPY ./watermark_reliability_release/requirements.txt /requirements2.txt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && pip install --no-cache-dir -r requirements2.txt

# Change ssh config and restart ssh
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# set environment variables
ENV NLTK_DATA /workspace/.cache/NLTK_DATA
ENV HF_DATASETS_CACHE /workspace/.cache/huggingface/datasets
ENV TRANSFORMERS_CACHE /workspace/.cache/huggingface/transformers