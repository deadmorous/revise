# About ReVisE
TODO

# Installing ReVisE

## Install prerequisites
TODO

## Clone, bootstrap, and build ReVisE

First of all, open a Linux terminal and switch to a directory of your choice, where ReVisE is to be installed. Then execute the following commands.

```{bash}
# Clone the ReVisE repository
# TODO: Change URL
git clone git@equares.ctmech.ru:stepan/revise.git
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
It might be necessary to set up envoronment variables in order to run ReVisE programs. To do so, create file `env.sh` in the `scripts/custom` subdirectory. This file, when exists, is sourced by other script that run ReVisE programs. For example, `env.sh` may be as follows:
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

# Configuring hardware resources
TODO

# Preparing datasets
TODO

# Measuring ReVisE performance
TODO
