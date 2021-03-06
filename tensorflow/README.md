# Tensorflow

## virtual env

```
mkvirtualenv --python=python3 learn-tensorflow

pip install -U pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Python requirements

I'm using python 3.5 or 3.6.

You may also need to install the extra things below:

```
# for matplotlib
sudo apt-get install python3-tk
```

## GPU

Follow instructions here: https://www.tensorflow.org/install/gpu
to set up GPU support.

VIA DOCKERFILE:

```
cd learn/tensorflow
docker build docker/tf2 -t tf2:latest
```

And run

```
nvidia-docker run -u $(id -u):$(id -g) -it --rm --name tf2 -v ~/code/learn/tensorflow:/app tf2:latest /bin/bash
```

or use run script:
```
./docker/run_tf2_docker.sh
```

Check if tensorflow is actually using GPU

```
python

# from python
import tensorflow as tf

print("TF version={}".format(tf.__version__))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
```
