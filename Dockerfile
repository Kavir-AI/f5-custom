FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/Kavir-AI/f5-custom"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace

RUN git clone https://github.com/Kavir-AI/f5-custom.git \
    && cd f5-custom \
    && git submodule update --init --recursive \
    && sed -i '8iimport sys\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))' src/third_party/BigVGAN/bigvgan.py \
    && pip install -e . --no-cache-dir \
    && pip install --no-cache-dir \
        celluloid>=0.2.0 \
        deepspeed>=0.12.4 \
        librosa>=0.10.1 \
        matplotlib>=3.8.1 \
        numpy>=1.26.2 \
        omegaconf>=2.3.0 \
        pandas>=2.1.3 \
        ptflops>=0.7.1.2 \
        rich>=13.7.0 \
        scipy>=1.11.4 \
        soundfile>=0.12.1 \
        torch>=2.1.1 \
        torchaudio>=2.1.1 \
        torchvision>=0.16.1 \
        tqdm>=4.66.1 \
        resampy>=0.4.2 \
        tabulate>=0.8.10 \
        gradio>=4.8.0

ENV SHELL=/bin/bash

WORKDIR /workspace/f5-custom
