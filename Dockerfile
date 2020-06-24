FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN mkdir /dy
WORKDIR /dy
EXPOSE 15015
EXPOSE 6007

ENV DOCKERNAME=dl_with_np

RUN apt update \
&& apt install -y python3 \
&& apt install -y python3-pip git \
&& pip3 install jupyter tqdm pillow pyyaml scipy 
RUN python3 -m pip install --upgrade pip

RUN pip3 install matplotlib
RUN pip3 install idx2numpy

ENTRYPOINT ["/bin/sh", "-c", "jupyter-notebook --no-browser --port 15015 --ip=0.0.0.0 --NotebookApp.token='kdy' --allow-root --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

