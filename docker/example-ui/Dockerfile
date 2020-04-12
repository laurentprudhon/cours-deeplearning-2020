FROM laurentprudhon/fastai2:fastai2-cpu-0.0.16

MAINTAINER Laurent Prud'hon

RUN mkdir ~/bears-ui
WORKDIR ~/bears-ui

COPY bears.pkl .
COPY bearsUI.ipynb .

EXPOSE 8080

CMD voila bearsUI.ipynb --port 8080 --no-browser
