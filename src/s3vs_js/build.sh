#!/bin/bash
set -e
npm install
node-gyp configure
node-gyp build

