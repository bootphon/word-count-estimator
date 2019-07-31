FROM python:3.7-slim-buster

COPY requirements.txt /

RUN pip install -r /requirements.txt
RUN apt-get update && apt-get install -y libsndfile1

COPY src/ /app
WORKDIR /app

VOLUME ../results

ENTRYPOINT ["python", "cli.py"]
