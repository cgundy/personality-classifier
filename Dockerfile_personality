FROM python:3.8-slim

WORKDIR /api
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY ./personality_classifier ./personality_classifier
COPY config.yml config.yml

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["personality_classifier.app:app", "--host", "0.0.0.0"]
