import logging
from model import Model


logger = logging.getLogger('model')
#self._model_path: str = model_dir / Path(model_type + '.pkl')
#self._tfidf_path: str = model_dir / 'tfidf_vec.pkl'


class Pipeline:
	def __init__(self, config, model_type):
		self.config = utils.get_config()
		self.model_type = model_type
		self.tfidf

	def get_tfidf(self, train=False):
		tfidf = TfidfFit(self.config['tfidf_params'], self.config['my_stop_words'])
		if train:
			logger.info("""Training tfidf.""")
			tfidf.train_tfidf()
			tifdf.save()
		else:
			logger.info("""Loading tfidf.""")
			tfidf.load()
		self.tfidf = tfidf

	@staticmethod
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

	def prep_data(df:pd.DataFrame) -> Tuple[pd.Series, pd.Series]:

		data_file = utils.file_handler()
		raw_data = pd.read_csv(data_file)

		logger.info("""Prepping data.""")
		df.columns = ['label', 'text']
		for index, row in df.iterrows():
			df.text[index] = _remove_types(df.text[index])
		self.label = df['label']
		self.text = df['text']

	def prep_features(self, train=False):
		if train:
			self.features = self.tfidf.fit_tfidf(self.text)
		else:
			self.features = self.tfidf.refit_tfidf(self.text)

	def train_model(self) -> None:
		model = Model(self.model_type, self.config)
		logger.info("Training {model_type} model.")
		model.train(self.features, self.labels)
		logger.info("Saving {model_type} model.")
		model.save()

	def score(self) -> str:
		model = Model(self.model_type)
		model.load('model')
		y_pred = model.predict(self.features)

	def get_accuracy(self) -> float:
		model = Model(self.model_type)
		model.load('accuracy')
		self.accuracy = model._accuracy


def train_pipeline(Pipeline, retrain_tfidf=False):
	Pipeline.get_tfidf(retrain_tfidf)
	Pipeline.prep_data()
	Pipeline.prep_features()
	Pipeline.train_model()

def score_pipeline(Pipeline):
	Pipeline.get_tfidf()
	Pipeline.prep_features()
	Pipeline.score()