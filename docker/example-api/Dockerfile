FROM laurentprudhon/fastai2:fastai2-cpu-0.0.16

MAINTAINER Laurent Prud'hon

RUN mkdir ~/bears-api
WORKDIR ~/bears-api

COPY bears.pkl .
COPY bears.py .
COPY bearsapi.py .

EXPOSE 8000

CMD ["uvicorn", "bearsapi:app", "--host", "0.0.0.0", "--port", "8000"]