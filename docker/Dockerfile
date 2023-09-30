FROM ubuntu:20.04
USER root

# install
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tar \
    bzip2 \
    vim \
    make \
    gcc \
    g++ \
    python3.8 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /root/tmp

# copy & unzip
ADD docker/lib/CRF++-0.58.tar.gz /root/tmp
ADD docker/lib/mecab-0.996.tar.gz /root/tmp
ADD docker/lib/mecab-jumandic-7.0-20130310.tar.gz /root/tmp
ADD docker/lib/cabocha-0.69.tar.bz2 /root/tmp
ADD docker/lib/jumanpp-1.02.tar.xz /root/tmp

# complie
WORKDIR /root/tmp/CRF++-0.58
RUN ./configure \
    && make \
    && make install

WORKDIR /root/tmp/mecab-0.996
RUN ./configure --with-charset=utf8 \
    && make \
    && make install \
    && echo '/usr/local/lib' >> /etc/ld.so.conf.d/local.conf \
    && ldconfig

WORKDIR /root/tmp/mecab-jumandic-7.0-20130310
RUN ./configure --with-charset=utf8 \
    && make \
    && make install

WORKDIR /root/tmp/cabocha-0.69
RUN ./configure --with-charset=UTF8 \
    && make \
    && make install \
    && ldconfig

WORKDIR /root/tmp/jumanpp-1.02.tar.xz
RUN ./configure --with-charset=utf8 \
    && make \
    && make install \
    && ldconfig

# set path for python
RUN rm -rf /root/tmp \
    && ln -s /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /usr/bin/python3 \
    && ln -s /usr/bin/python3.8 /usr/bin/python3

# copy
COPY docker/mecabrc /usr/local/etc/mecabrc
COPY poetry.lock .
COPY pyproject.toml .

# pip install
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir poetry

WORKDIR /code
