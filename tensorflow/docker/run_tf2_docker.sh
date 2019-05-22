#!/usr/bin/env bash

nvidia-docker run -u $(id -u):$(id -g) -it --rm --name tf2 -v ~/code/learn/tensorflow:/app tf2:latest /bin/bash
