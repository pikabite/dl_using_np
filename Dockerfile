FROM ubuntu:20.04

RUN mkdir /workspace
WORKDIR /workspace
EXPOSE 15015

ENV DOCKERNAME=dl_using_np

RUN apt update \
&& apt install -y python3 \
&& apt install -y python3-pip git \
&& pip3 install jupyter tqdm pillow pyyaml scipy opencv-python opencv-contrib-python
RUN python3 -m pip install --upgrade pip

RUN apt install libgl1-mesa-glx

RUN pip3 install matplotlib
RUN pip3 install idx2numpy

ARG UNAME=kdy
ARG UID=$UID
ARG GID=$GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME

USER kdy


ENTRYPOINT ["/bin/sh", "-c", "jupyter-notebook --no-browser --port 15015 --ip=0.0.0.0 --NotebookApp.token='kdy' --allow-root --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

