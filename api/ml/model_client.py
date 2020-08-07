
with open('config.yml') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)

logger = logging.getLogger('model')

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


def train_model(model_type: str, train_tfidf=False) -> None:
    tfidf = TfidfFit(config['tfidf_params'], config['my_stop_words'])
    if train_tfidf:
        logger.info("""Training tfidf.""")
        tfidf.train_tfidf()
    else:
        logger.info("""Loading tfidf.""")
        tfidf.load()
    
    logger.info("""Prepping data.""")
    data_file = utils.file_handler()
    raw_data = pd.read_csv(data_file)
	label, text = prep_data(raw_data)
	logger.info("Fitting tfidf vectorizor")
	X = tfidf.fit_tfidf(text)

    model = Model(model_type, config)
	y = label
	logger.info("Training {model_type} model.")
	model.train(X, y)
	logger.info("Saving {model_type} model.")
	model.save()

def score(text_input: List[str], model_type: str) -> str:
	model = Model(model_type)
	model.load()
    features = model.refit_tfidf(text_input)
	y_pred = model.predict(input)
	return y_pred

def get_accuracy(model_type: str) -> float:
	model = Model(model_type)
	model.load()
	return model._accuracy

		self._model_path: str = model_dir / Path(model_type + '.pkl')
		self._tfidf_path: str = model_dir / 'tfidf_vec.pkl'


