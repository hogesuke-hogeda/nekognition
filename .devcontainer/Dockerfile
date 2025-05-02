FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

RUN apt-get update && \
    apt-get install -y \
    locales \
    bash-completion \
    iputils-ping \
    net-tools \
    jq \
    fzf \
    vim \
    git \
    curl \
    wget \
    less \
    groff

# ロケール設定
RUN echo 'ja_JP.UTF-8 UTF-8' >> /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=ja_JP.UTF-8
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
