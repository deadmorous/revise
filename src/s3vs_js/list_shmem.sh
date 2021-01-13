#!/bin/bash
ipcs |grep " 666 " |sed -r "s/^[^ ]+ ([0-9]+).*$/\\1/"

