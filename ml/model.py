import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

types = ['infj', 'infp', 'intj', 'intp', 'istj', 'istp', 'isfp',
		'isfj', 'enfj', 'enfp', 'entj', 'entp', 'estj', 'estp', 'esfp', 'esfj']
my_stop_words = text.ENGLISH_STOP_WORDS.union(['ni', 'ti', 'ne', 'te', 'se'])


def remove_types(text):
	for ptype in types:
		if ptype in text or ptype.upper() in text:
			text = text.replace(ptype, "")
			text = text.replace(ptype+'s', "")
			text = text.replace(ptype+'\'s', "")
			text = text.replace(ptype.upper(), "")
			text = text.replace(ptype.upper()+'s', "")
			text = text.replace(ptype.upper()+'\'s', "")
	return text

def prep_data(df):
	df.columns = ['label', 'text']
	for index, row in df.iterrows():
		df.text[index] = remove_types(df.text[index])
	return df['label'], df['text']

class Model:
	def __init__(self, model_type):
		self.model_type = model_type
		self._model_path = 'ml/models/' + str(model_type) + '.pkl'
		self._tfidf_path = 'ml/models/tfidf_vec.pkl'
		
	def fit_tfidf(self, text):
		tfidf = TfidfVectorizer(stop_words=my_stop_words,
								sublinear_tf=True, min_df=10, ngram_range=(1, 2))
		features = tfidf.fit_transform(text).toarray()
		self._tfidf = tfidf
		return features

# Todo: move parameters to config file
	def refit_tfidf(self, text):
		tf1_new = TfidfVectorizer(stop_words=my_stop_words, sublinear_tf=True, min_df=0, 
								ngram_range=(1, 2), vocabulary=self._tfidf.vocabulary_)
		features = tf1_new.fit_transform(text).toarray()
		return features

	def train(self, X, y):
		if self.model_type == 'LogisticRegression':
	   		model = LogisticRegression(random_state=0, solver='lbfgs', class_weight=None).fit(X, y)
		elif self.model_type == 'RandomForest':
			model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X, y)
		elif self.model_type == 'SVC':
			model = SVC(C = 1, gamma = 1, kernel = 'rbf')
		elif self.model_type == 'MultinomialNB':
			model = MultinomialNB()
		self._model = model

	def calc_model_accuracy(self, X, y):
		return self._model.score(X,y)

	def save(self):
		if self._tfidf is not None:
			filename = 'models/tfidf_vec.pkl'
			with open(filename, 'wb') as infile:
				pickle.dump(self._tfidf, infile)
		else:
			raise TypeError("The tfidf vector is not trained yet, use .fit_tfidf() before saving")
		if self._model is not None:
			filename = 'models/' + self.model_type+'.pkl'
			with open(filename, 'wb') as infile:
				pickle.dump(self._model, infile)
		else:
			raise TypeError("The model is not trained yet, use .train() before saving")

	def load(self):
		try:
 		  	self._tfidf = np.load(self._tfidf_path, allow_pickle=True)
		except:
			raise TypeError(
			"The tfidf vector is not trained yet, use .fit_tfidf() before loading")
		try:
			self._model = np.load(self._model_path, allow_pickle=True)
		except:
			raise TypeError(f"The model is not trained yet, use .train() before loading. {self._model_path}")

	def predict(self, text_input):
		features = self.refit_tfidf(text_input)
		prediction = self._model.predict(features)
		return prediction[0]


def retrain(model_type):
	model = Model(model_type)
	raw_data = pd.read_csv('mbti_1.csv')
	label, text = prep_data(raw_data)
	X = model.fit_tfidf(text)	
	y = label
	model.train(X, y)
	model.save()

def score(input=['Hello my name is carly'], model_type='LogisticRegression'):
	model = Model(model_type)
	model.load()
	y_pred = model.predict(input)
	print(y_pred)
	return y_pred

