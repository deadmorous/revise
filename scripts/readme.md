# Scripts directory

The directory contains the `env.sh` bash script used to set up environment for running components of the system.

The directory also contains the `custom` subdirectory, in which all `*.sh` files are not tracked by git. Custom scripts may be added to the `custom` subdirectory to adjust the enviromnent for a specific user. For example, there may be the `custom/env.sh` script file. If that file exists, it is sourced from `env.sh` Other files may exist too, optionally used by other scripts
