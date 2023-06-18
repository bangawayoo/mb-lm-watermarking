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
    unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Change ssh config and restart ssh
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# set environment variables
ENV NLTK_DATA /workspace/.cache/NLTK_DATA
ENV HF_DATASETS_CACHE /workspace/.cache/huggingface/datasets
ENV TRANSFORMERS_CACHE /workspace/.cache/huggingface/transformers