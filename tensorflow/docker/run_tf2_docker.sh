#!/usr/bin/env bash

# Basic
#nvidia-docker run -u $(id -u):$(id -g) -it --rm --name tf2 -v ~/code/learn/tensorflow:/app tf2:latest /bin/bash

# With display for matplotlib

nvidia-docker run -u $(id -u):$(id -g) -it --rm --name tf2 \
   -v ~/code/learn/tensorflow:/app \
   --env="DISPLAY" \
   --workdir=/app \
   -v "$PWD":/app \
   -v "/etc/group:/etc/group:ro" \
   -v "/etc/passwd:/etc/passwd:ro" \
   -v "/etc/shadow:/etc/shadow:ro" \
   -v "/etc/sudoers.d:/etc/sudoers.d:ro" \
   -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   tf2:latest /bin/bash