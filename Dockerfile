FROM python:3.8-slim

WORKDIR /api

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY setup.py setup.py
RUN pip install .

COPY ./personality_classifier ./personality_classifier
COPY config.yml config.yml
COPY ./tests ./tests
