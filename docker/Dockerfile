FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get install -y libglib2.0-0 && apt-get clean

RUN apt-get install -y wget htop byobu git gcc g++ vim libsm6 libxext6 libxrender-dev lsb-core

RUN cd /root && wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh

RUN cd /root && bash Anaconda3-2020.07-Linux-x86_64.sh -b -p ./anaconda3

RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda install -y pytorch==1.5.0 torchvision cudatoolkit=10.2 -c pytorch"

RUN bash -c "/root/anaconda3/bin/conda init bash"

WORKDIR /root
RUN mkdir code
WORKDIR code

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda activate base && cd detectron2 && python setup.py build develop"

RUN git clone https://github.com/aim-uofa/AdelaiDet.git adet

WORKDIR adet
RUN bash -c "source /root/anaconda3/etc/profile.d/conda.sh && conda activate base && python setup.py build develop"

RUN rm /root/Anaconda3-2020.07-Linux-x86_64.sh
