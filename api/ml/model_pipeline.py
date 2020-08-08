import logging
import pandas as pd
from typing import List, Set, Dict, Tuple, Optional, Union, Any

from .model import MyClassifier
from .utilities import get_config, file_handler, save, load

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


config = get_config()
logger = logging.getLogger('model')


def _remove_types(text: str) -> str:
	for ptype in config['types']:
		if ptype in text or ptype.upper() in text:
			text = text.replace(ptype, "")
			text = text.replace(ptype+'s', "")
			text = text.replace(ptype+'\'s', "")
			text = text.replace(ptype.upper(), "")
			text = text.replace(ptype.upper()+'s', "")
			text = text.replace(ptype.upper()+'\'s', "")
	return text


def train(model_type: str) -> None:
	data_file = file_handler('data')
	df = pd.read_csv(data_file)
	df.columns = ['label', 'text']
	for index, row in df.iterrows():
		df.text[index] = _remove_types(df.text[index])
	X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=0.2, random_state = 28)

	pipe = Pipeline([
		('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(config['my_stop_words']), 
									**config['model_parameters']['tfidf'])),
		('clf', MyClassifier(config, model_type))])

	pipe.fit(X_train, y_train)
	save(pipe, 'pipeline', model_type)
	save(pipe.score(X_test, y_test), 'accuracy', model_type)

def predict(text_input: List[str], model_type: str) -> str:
	pipe = load('pipeline', model_type)

	#pipe = Pipeline([
	#	('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(config['my_stop_words']), 
	#									**config['model_parameters']['tfidf'], vocabulary=tfidf.vocabulary_))])

#	pipe.fit_transform([text_input])
#	pipe.steps.append(('clf', model))
	return pipe.predict(text_input)

def get_accuracy(model_type: str) -> float:
	return load('accuracy', model_type)

#train('RandomForest')
#print(predict(['hello my name is Carly'],'RandomForest'))
#print(get_accuracy('RandomForest'))
