FROM python:3.7-slim-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    sox

COPY ./ /
WORKDIR /

ENTRYPOINT ["python", "./cli.py"]
