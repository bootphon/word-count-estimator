FROM python:3.7-slim-buster

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r ./requirements.txt
RUN apt-get clean && apt-get update && apt-get install -y \
    libsndfile1 \
    sox

RUN mkdir ./data ./results

COPY ./ ./

ENTRYPOINT ["python", "cli.py"]
CMD ["predict", "data/", "data/", "results/output.csv", "opensmileSad", \
     "-w", "models/word_count_estimator/adapted_model.pickle"]
