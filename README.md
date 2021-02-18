# About ReVisE
TODO

# Installing ReVisE

## Install prerequisites

### Supported Linux distributions

- Ubuntu (20.10)
- Debian Testing (10)
- Arch Linux

### Software requirements

 - cmake >= 3.18.1
 - git
 - clang >= 9.0 / gcc >= 7.4
 - Qt >= 5.10 including QtSVG, QtWidgets
 - Node.js
 - npm
 - D language
 - CUDA Toolkit >= 11.1

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Debian&target_version=10&target_type=deblocal



## Clone, bootstrap, and build ReVisE

First of all, open a Linux terminal and switch to a directory of your choice, where ReVisE is to be installed. Then execute the following commands.

```{bash}
# Clone the ReVisE repository
# TODO: Change URL
git clone https://github.com/deadmorous/revise.git
cd revise

# Optionally set up Qt, e.g. 
# export QT_SELECT=qt5-11-2
#
# Run bootstrap.sh
./bootstrap.sh

# At this point, third-party repositories should be
# cloned in third_parties, and there must appear
# directories build and dist.

# Build ReViSe
./build.sh
```

## Set up custom environment
It might be necessary to set up environment variables in order to run ReVisE programs. To do so, create file `env.sh` in the `scripts/custom` subdirectory. This file, when exists, is sourced by other script that run ReVisE programs. For example, `env.sh` may be as follows:
```{bash}
#!/bin/bash

export QT_SELECT=qt5-11-2
export LD_LIBRARY_PATH=$(qmake -query QT_INSTALL_LIBS):$HOME/oss/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
The script might differ on your system or might be not necessary at all.

## Generate example datasets
Once ReVisE is built, example datasets need to be created in order to feed something to the visualizer. To generate those examples, run
```{bash}
./data/make_revise_web_demo.sh
```
This might take a while. For example, on a desktop system with Intel i7-8700 CPU @ 3.20GHz (6 cores), 16 GB RAM and a regular HDD it takes about 10 minutes.

Notice that the datasets generated at this step occupy about 10 Gb of disk space.

## Run ReVisE web server
To start the web server, run
```{bash}
./src/webserver/start.sh 
```
When started for the first time, the script downloads and installs necessary dependencies and builds two ReVisE modules that interoperate with node.js. That process takes a few seconds. When the server starts, it displays the message
```
Starting web server
```
meaning that the server is ready.

Once the server is ready, open your Web browser and enter the following URL in the address bar:
```
http://localhost:3000/
```
You should see TeVisE GUI in the browser and be able to choose among example problems.

# Docker


## Setup NVIDIA driver and runtime

Verify the installation with the command nvidia-smi.
If NVIDIA driver is not pre-installed with your Linux distribution, 
you can install it with sudo apt install nvidia-XXX (XXX is the version, the newest one is 440) or 
download the appropriate NVIDIA driver and execute the binary as sudo.

Install NVIDIA container runtime:

```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```


Restart Docker:

```
sudo systemctl stop docker
sudo systemctl start docker
```

Now you are ready to run the application in Docker.



## Building image by yourself.

If you want to build a image by yourself you need to run this command in repo:
```
docker build -t <name>:<tag> .
```


## Run our docker's image

Download docker's image to your machine:

Cuda 11.2
```
docker pull deadmorous/revise:v1
```

Cuda 11.1
```
docker pull deadmorous/revise:cuda11.1
```


Run docker image and log into it:

```
docker run --gpus all -ti --rm -v <HOME_PATH>:<HOME_PATH> -p 3000:3000 -p 1234:1234 deadmorous/revise:v1  /bin/bash
```

You can specify the number of GPUs and even the specific GPUs with the --gpus flag. 

You can find more information here: https://docs.docker.com/engine/reference/commandline/run/#access-an-nvidia-gpu

After that, you can run revise's scripts and start webserver as usual. 


# Configuring hardware resources
TODO

# Preparing datasets
TODO

# Measuring ReVisE performance
TODO
