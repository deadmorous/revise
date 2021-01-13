node.js binding for s3dmm render server
=======================================

Pre-requisites
--------------

1. node.js
2. npm
3. python 3.x (3.7.6 works)
4. node-gyp, to install do this:<br/>
   `$ sudo npm install -g node-gyp`
5. s3dmm release binary files (see below how to build).

Building s3dmm release binaries
-------------------------------

```bash
$ this_dir=$PWD
$ s3dmm_dir=$(realpath ../..)
$ build_dir=$s3dmm_dir/builds/s3dmm/release
$ cd $s3dmm_dir
$ ./bootstrap.sh
$ mkdir -p $build_dir && cd $build_dir
$ # to see available Qt versions: qtchooser -l
$ # to select particular Qt version: export QT_SELECT=...
$ cmake -DS3DMM_ENABLE_GUI=ON -DS3DMM_ENABLE_REN=ON -DS3DMM_REAL_TYPE=float ../../..
$ # If you have a recent NVidia GPU and CUDA 10 Toolkit installed, also add -DS3DMM_ENABLE_CUDA=ON
$ make # or make -jN, where N equals the number of CPU cores
$ cd $this_dir
```

Building and running
--------------------

To build the project, do this
```bash
$ npm install
$ node-gyp configure
$ node-gyp build
```

The same commands are in file `build.sh`.

To run the `index.js` example JavaScript file using the module, do this
```bash
$ ./start.sh
```

Notice that the script `start.sh` sets up environment variables and then runs
```
node index.js
```

The environment variables that need to be set up are

- `s3vs_binary_dir` - the directory containing s3dmm release binaries
- `LD_LIBRARY_PATH` - must contain path to s3dmm third party release binaries

See file `node_env.sh`, where those variables are set.

Reference
---------

The source code for this project (see the `src` subdirectory) is based on this tutorial:

[A simple guide to load C/C++ code into Node.js JavaScript Applications](https://itnext.io/a-simple-guide-to-load-c-c-code-into-node-js-javascript-applications-3fcccf54fd32)
