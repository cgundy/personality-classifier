FROM python:3.8-slim

WORKDIR /api
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY ./api ./api
COPY config.yml config.yml

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["personality-classifier.app:app", "--host", "0.0.0.0"]
