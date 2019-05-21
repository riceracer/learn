# learn
tutorials and stuff

## Ubuntu

### install docker

Follow directions here: https://docs.docker.com/install/linux/docker-ce/ubuntu/

also do this for using the docker user group: https://docs.docker.com/install/linux/linux-postinstall/

### install mkvirtualenv

I used this for installing the wrapper with python3 on ubuntu 18.04

```
sudo apt-get install python3-pip
mkdir ~/.virtualenvs

pip3 install virtualenv
sudo -H pip3 install virtualenvwrapper

# add to ~/.bashrc
export WORKON_HOME=~/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=python3
source /usr/local/bin/virtualenvwrapper.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

## install nvidia-docker

Instructions here: https://github.com/NVIDIA/nvidia-docker
* install version 2.0 or greater
