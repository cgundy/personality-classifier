import os
import yaml
import numpy as np
import pandas as pd
import pickle
import mypy
import logging
from typing import List, Set, Dict, Tuple, Optional, Union
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

with open('config.yml') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

data_file = Path(__file__).parent.parent / 'data/mbti_1.csv'
model_dir = Path(__file__).parent.parent / 'ml/models' 

def remove_types(text: str) -> str:
	for ptype in config['types']:
		if ptype in text or ptype.upper() in text:
			text = text.replace(ptype, "")
			text = text.replace(ptype+'s', "")
			text = text.replace(ptype+'\'s', "")
			text = text.replace(ptype.upper(), "")
			text = text.replace(ptype.upper()+'s', "")
			text = text.replace(ptype.upper()+'\'s', "")
	return text

def prep_data(df:pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
	df.columns = ['label', 'text']
	for index, row in df.iterrows():
		df.text[index] = remove_types(df.text[index])
	return df['label'], df['text']

class Model:
	def __init__(self, model_type:str):
		self.model_type: str  = model_type
		self._model_path: str = model_dir / Path(model_type + '.pkl')
		self._tfidf_path: str = model_dir / 'tfidf_vec.pkl'
		self._tfidf: TfidfVectorizer
		self._model: Union[LogisticRegression, RandomForestClassifier,
		SVC, MultinomialNB]
		self._accuracy: float

	def fit_tfidf(self, text_input: List[str]) -> np.array:
		tfidf = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(config['my_stop_words']), **config['tfidf_params'])
		features = tfidf.fit_transform(text_input).toarray()
		self._tfidf = tfidf
		return features

	def refit_tfidf(self, text_input: List[str]) -> np.array:
		tf1_new = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(config['my_stop_words']), **config['tfidf_params'], vocabulary=self._tfidf.vocabulary_)
		features = tf1_new.fit_transform(text_input).toarray()
		return features

	def train(self, X: np.array, y: str) -> None:
		model = None
		if self.model_type == 'LogisticRegression':
	   		model = LogisticRegression(**config['lr_params']).fit(X, y)
		elif self.model_type == 'RandomForest':
			model = RandomForestClassifier(**config['rf_params']).fit(X, y)
		elif self.model_type == 'SVC':
			model = SVC(**config['svc_params']).fit(X, y)
		elif self.model_type == 'MultinomialNB':
			model = MultinomialNB()
		self._model = model
		
	def _calc_model_accuracy(self, X: np.array, y: str) -> float:
		self._accuracy = self._model.score(X,y)

	def save(self):
		if self._tfidf is not None:
			with open(self._tfidf_path, 'wb') as infile:
				pickle.dump(self._tfidf, infile)
		else:
			raise TypeError("The tfidf vector is not trained yet, use .fit_tfidf() before saving")
		if self._model is not None and self._accuracy is not None:
			with open(self._model_path, 'wb') as infile:
				pickle.dump([self._model, self._accuracy], infile)
		else:
			raise TypeError("The model is not trained yet, use .train() before saving")

	def load(self):
		try:
 		  	self._tfidf = np.load(self._tfidf_path, allow_pickle=True)
		except:
			raise TypeError(
			"The tfidf vector is not trained yet, use .fit_tfidf() before loading")
		try:
			self._model = np.load(self._model_path, allow_pickle=True)[0]
			self._accuracy = np.load(self._model_path, allow_pickle=True)[1]

		except:
			raise TypeError(f"The model is not trained yet, use .train() before loading. {self._model_path}")

	def predict(self, text_input: List[str]) -> str:
		features = self.refit_tfidf(text_input)
		prediction = self._model.predict(features)
		return prediction[0]

#Todo: add logging
def retrain(model_type: str) -> None:
	model = Model(model_type)
	raw_data = pd.read_csv(data_file)
	label, text = prep_data(raw_data)
	X = model.fit_tfidf(text)	
	y = label
	model.train(X, y)
	model._calc_model_accuracy(X, y)
	model.save()

def score(input: List[str], model_type: str) -> str:
	model = Model(model_type)
	model.load()
	y_pred = model.predict(input)
	return y_pred

def get_accuracy(model_type: str) -> float:
	model = Model(model_type)
	model.load()
	return model._accuracy

#retrain('LogisticRegression')	
