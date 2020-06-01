FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update \
    && apt-get -y install \
        python3-pip \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && apt-get -y install \
        python3.6 \
        python3.6-dev \
    && add-apt-repository ppa:ubuntugis/ppa \
    && apt-get -y update \
    && apt-get -y install \
        gdal-bin \
        python3-rtree \
        libgdal-dev \
        wget \
        unzip \
        git \
    && export CPLUS_INCLUDE_PATH=/usr/include/gdal \
    && export C_INCLUDE_PATH=/usr/include/gdal \
    && pip3 install --upgrade pip \
    && pip3 install \
        GDAL==2.2.2 \
        sentinelsat \
        folium \
        pandas \
        matplotlib \
        shapely \
        rasterio \
        geopandas \
        descartes \
        fiona \
        Rtree==0.8.3 \
        earthengine-api \
        jupyter

EXPOSE 8888