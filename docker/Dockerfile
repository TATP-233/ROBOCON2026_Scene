FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    software-properties-common \
    libgl1-mesa-dev \
    libglew-dev \ 
    libegl1-mesa-dev \ 
    libgles2-mesa-dev \ 
    libnvidia-egl-wayland1 \
    libosmesa6-dev \
    xvfb \ 
    libx11-6 \ 
    libxext6 \ 
    libglfw3-dev \ 
    libglu1-mesa-dev \ 
    libglm-dev \
    pkg-config \
    --no-install-recommends

# Install Python 3.10
RUN apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip

RUN python3 -m pip install --upgrade setuptools pip build

RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
   echo '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get install -y \
    wget \
    sudo \
    x11-apps \
    mesa-utils \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /root

CMD ["/bin/bash"]