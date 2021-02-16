FROM debian:testing-slim as base
MAINTAINER ReVisE https://github.com/deadmorous/revise
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.2
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH $PATH:$CUDA_HOME/bin
ENV CUDA_PATH $CUDA_HOME

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-minimal-build-${CUDA_VERSION} \
    cuda-nvprof-${CUDA_VERSION} \
    && ln -s cuda-${CUDA_VERSION} /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

FROM base as builder

RUN apt-get update \
    && apt-get install -y build-essential \
        cmake \
        git \
        clazy \
        clang \
        clang-tidy \
        libstdc++-10-dev \
        gettext \
        software-properties-common \
    && ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/libstdc++.so \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y libboost-all-dev \
        qtbase5-dev \
        qttools5-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Project specific libraries
#############

RUN apt-get update \
    && apt-get install -y \
        libqt5websockets5-dev \
        libcairo2-dev \
        pkg-config \
        libopencv-dev \
        libqt5svg5-dev \
        libglew-dev \
        npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* ./*.deb

COPY . /src
WORKDIR /src

RUN ./bootstrap.sh --with-tsv-util \
    && ./build.sh

FROM base as runtime

WORKDIR /opt/revise/

COPY --from=builder /src/src ./src
COPY --from=builder /src/data ./data
COPY --from=builder /src/include ./include
COPY --from=builder /src/scripts ./scripts
COPY --from=builder /src/dist ./dist
COPY --from=builder /src/builds ./builds

RUN apt-get update \
    && apt-get install -y \
        libqt5websockets5 \
        libcairo2-dev \
        pkg-config \
        libqt5svg5 \
        libopencv-dev \
        libglew-dev \
        npm \
        libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* ./*.deb

EXPOSE 3000

#CMD ["src/webserver/start.sh"]
