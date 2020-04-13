FROM continuumio/miniconda3:4.8.2-alpine

MAINTAINER Laurent Prud'hon

USER 10151
ENV PATH "/opt/conda/bin:/opt/conda/condabin:/user/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Basic Python librairies : config, images, http
RUN conda install -y pyyaml pillow requests && conda clean -y -a
# API framework + http server
RUN conda install -y -c conda-forge fastapi uvicorn python-multipart && conda clean -y -a
# Numeric computing librairies
RUN conda install -y numpy pandas pyarrow && conda clean -y -a
# Scientific computing librairies
RUN conda install -y scikit-learn scipy && conda clean -y -a
# Deep learning computing library : PyTorch - CPU version
RUN conda install -y -c pytorch pytorch torchvision cpuonly && conda clean -y -a
# UI framework + notebook server
RUN conda install -y -c conda-forge jupyter ipykernel matplotlib voila && conda clean -y -a
RUN jupyter serverextension enable voila --sys-prefix
# Natural language processing library
RUN conda install -y -c conda-forge spacy && conda clean -y -a
# High level deep learning library : fastai2 
RUN pip install fastcore fastprogress fastai2

CMD [ "sh", "--login", "-i" ]