#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

source node_env.sh $1

set -e

if [ ! -f ./../s3vs_js/build/Release/s3vs_js.node ]; then
	cd ./../s3vs_js
	./build.sh
	cd ${SCRIPT_DIR}
fi

if [ ! -f ./../ws_sendframe_js/build/Release/ws_sendframe_js.node ]; then
	cd ./../ws_sendframe_js
	./build.sh
	cd ${SCRIPT_DIR}
fi

cd $SCRIPT_DIR
mkdir -p logs

[ ! -d "node_modules" ] && npm install

echo Starting web server
node bin/www

