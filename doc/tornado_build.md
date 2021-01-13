Building and running s3dmm on Tornado
=====================================

Prerequisites
-------------

- Add some environment variables to your `~/.bashrc` file:

```
export GROUP_SHARED_DIR=/home/immtktm/b
export PATH=$GROUP_SHARED_DIR/bin:$PATH:$GROUP_SHARED_DIR/qt/5.11.2/gcc_64/bin
export CMAKE_MODULE_PATH=$GROUP_SHARED_DIR/qt/5.11.2/gcc_64/lib/cmake
export GROUP_SHARED_LIB_DIR=$GROUP_SHARED_DIR/lib
export GLEW_LIBRARY_DIR=$GROUP_SHARED_LIB_DIR
```

- Install nvm locally

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
```

- Install node.js locally

```
nvm install node
```

- Load some slurm modules

```
module load compiler/gcc/7     #для CUDA 9.2
```

- Setup SSH keys for using git@equares.ctmech.ru.

Cloning
-------
```
git clone git@equares.ctmech.ru:stepan/s3dmm.git
```

Building
--------
```
cd s3dmm
module load nvidia/cuda/9.2 compiler/gcc/7
./bootstrap.sh
cd builds

mkdir -p s3dmm/release
cd s3dmm/release
cmake ../../.. -DGLEW_LIBRARY_DIR=/home/immtktm/b/lib -DS3DMM_ENABLE_GUI=ON -DS3DMM_ENABLE_REN=ON -DCMAKE_BUILD_TYPE=Release
make -j12

module load compiler/gcc/7
cmake ../../.. -DGLEW_LIBRARY_DIR=/home/immtktm/b/lib -DS3DMM_ENABLE_GUI=ON -DS3DMM_ENABLE_REN=ON -DS3DMM_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j12
module load compiler/gcc/9

cd ../../../src
cd s3vs_js
npm install
#./build.sh
cd ../ws_sendframe_js
./build.sh
cd ../webserver
npm install
```

Running
-------

### Generate test dataset
```
cd data/synthetic-tests/sphere
# source ../../env.sh
export PATH=$PATH:$(realpath ../../../builds/s3dmm/release/bin)
./make_data.sh
```

### Run webserver
TODO