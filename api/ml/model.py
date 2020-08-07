import numpy as np
import pandas as pd
import pickle
import mypy
import logging
from typing import List, Set, Dict, Tuple, Optional, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import utilities as utils


class TfidfFit:
	def __init__(self, tfidf_params: List[Any], my_stop_words: List[str]=None):
		self._tfidf_params = tfidf_params
		self._my_stop_words = my_stop_words
		self._tfidf: TfidfVectorizer
		self._filepath = utils.file_handler('models','tfidf')

	#Todo: how to ensure that this is not run if tdifdf exists
	def train_tfidf(self):
		my_stop_words = self._my_stop_words
		if not my_stop_words:
			my_stop_words = []
		tfidf = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(my_stop_words), 
								**self._tfidf_params)
		return tfidf

	#Todo: ensure that _tfidf has been loaded
	def fit_tfidf(self, text_input: List[str]) -> np.array:
		features = self._tfidf.fit_transform(text_input).toarray()
		return features

	def refit_tfidf(self, text_input: List[str]) -> np.array:
		tf1_new = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS.union(self._my_stop_words), 
									**self._tfidf_params, vocabulary=self._tfidf.vocabulary_)
		features = tf1_new.fit_transform(text_input).toarray()
		return features

	def save(self):
		if self._tfidf is not None:
			with open(self._filepath, 'wb') as infile:
				pickle.dump(self._tfidf, infile)
		else:
			raise TypeError("The tfidf vector is not trained yet, use .fit_tfidf() before saving")

	def load(self):
		try:
 		  	self._tfidf = np.load(self._filepath, allow_pickle=True)
		except:
			raise TypeError(
			"The tfidf vector is not trained yet, use .fit_tfidf() before loading")


class Model:
	def __init__(self, model_type:str, config):
		self.model_type: str  = model_type
		self._model_config = config
		self._model: Union[LogisticRegression, RandomForestClassifier,
		SVC, MultinomialNB]
		self._accuracy: float
		self._model_path = utils.file_handler('models', model_type)
		self._accuracy_path = utils.file_handler('accuracy',model_type)

	def train(self, X: np.array, y: str) -> None:
		model = None
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 28)
		if self.model_type == 'LogisticRegression':
	   		model = LogisticRegression(**self._model_config['lr_params']).fit(X_train, y_train)
		elif self.model_type == 'RandomForest':
			model = RandomForestClassifier(**self._model_config['rf_params']).fit(X_train, y_train)
		elif self.model_type == 'SVC':
			model = SVC(**self._model_config['svc_params']).fit(X_train, y_train)
		self._model = model	
		self._accuracy = self._model.score(X_test,y_test)

	def save(self):
		if self._model is not None and self._accuracy is not None:
			with open(self._model_path, 'wb') as infile:
				pickle.dump(self._model, infile)
			with open(self._accuracy_path, 'wb') as infile:
				pickle.dump(self._accuracy, infile)
		else:
			raise TypeError("The model is not trained yet, use .train() before saving")

	def load(self, request):
		try:
			if request=='model':
				self._model = np.load(self._model_path, allow_pickle=True)
			elif request == 'accuracy':
				self._accuracy = np.load(self._accuracy_path, allow_pickle=True)
		except:
			raise TypeError(f"The model is not trained yet, use .train() before loading. {self._model_path}")

	def predict(self, features: np.array) -> str:
		prediction = self._model.predict(features)
		return prediction[0]


