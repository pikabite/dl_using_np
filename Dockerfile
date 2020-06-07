FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN mkdir /dy
WORKDIR /dy
EXPOSE 12015
EXPOSE 6007

ENV DOCKERNAME=dl_pa2

RUN apt update \
&& apt install -y python3 \
&& apt install -y python3-pip git \
&& pip3 install jupyter tqdm pillow pyyaml scipy \
&& apt install -y libsm6 libxext6 libxrender1 libfontconfig1 \
&& pip3 install opencv-python opencv-contrib-python 
RUN pip3 install -U git+https://github.com/albu/albumentations
RUN python3 -m pip install --upgrade pip

RUN pip3 install matplotlib
RUN pip3 install idx2numpy

ENTRYPOINT ["/bin/sh", "-c", "jupyter-notebook --no-browser --port 12015 --ip=0.0.0.0 --NotebookApp.token='kdy' --allow-root --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

