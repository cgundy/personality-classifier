from .ml.model import *
import pytest
from pathlib import Path

#Todo: create test data and move to fixture
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
data_file = 'api/data/mbti_1.csv'
raw_data = pd.read_csv(data_file)

model_type = 'LogisticRegression'

with open('config.yml') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

def test_prep_data():
	label, text = prep_data(raw_data)
	assert all([x.lower() in config['types'] for x in label])
	assert all([isinstance(x, str) for x in text])

def test_tifdf():
	model=Model(model_type)
	label, text = prep_data(raw_data)
	X = model.fit_tfidf(text)
	assert isinstance(X, np.ndarray)
	assert isinstance(model._tfidf, TfidfVectorizer)

def test_score():
	result = score(['I am an introvert who enjoys coding and rock climbing'], model_type)
	assert result.lower() in config['types']

def test_accuracy():
	accuracy = get_accuracy(model_type)
	print(accuracy)
	assert isinstance(accuracy, float)



