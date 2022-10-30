IMAGE="personality-classifier"
TAG="latest"
DOCKER_RUN=docker run $(IMAGE):$(TAG)
DOCKER_RUN_APP=docker run -p 8000:8000 $(IMAGE):$(TAG)

serve: build
	$(DOCKER_RUN_APP) uvicorn personality_classifier.app:app --host 0.0.0.0 --port 8000

build: 
	docker build -t $(IMAGE):$(TAG) .

lock:
	pipenv lock

requirements.txt: lock
	pipenv requirements > requirements.txt

train: build
	$(DOCKER_RUN) train_model --model_type="LogisticRegression"

lint: build
	$(DOCKER_RUN) mypy personality_classifier

test: build
	$(DOCKER_RUN) pytest tests

.PHONY: serve build lock requirements.txt train lint test
