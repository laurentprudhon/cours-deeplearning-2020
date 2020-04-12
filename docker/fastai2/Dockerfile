FROM continuumio/miniconda3:4.8.2-alpine

MAINTAINER Laurent Prud'hon

# Basic Python librairies : config, images, http
RUN conda install -y pyyaml pillow requests
# API framework + Http server
RUN conda install -y conda fastapi uvicorn python-multipart
# Numeric computing librairies
RUN conda install -y numpy pandas pyarrow 
# Scientific computing librairies
RUN conda install -y scikit-learn scipy
# Deep learning computing library : PyTorch - CPU version
RUN conda install -y pytorch torchvision cpuonly -c pytorch
# Data science dev environment
RUN conda install -y jupyter ipykernel matplotlib voila -c conda-forgejupyter 
RUN jupyter serverextension enable voila --sys-prefix
# Natural language processing library
RUN conda install -y -c conda-forge spacy
# High levek deep learning library : fastai2 
RUN pip install  fastcore fastprogress fastai2

CMD [ "sh", "--login", "-i" ]