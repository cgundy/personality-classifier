import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Todo: move parameters to config file
def fit_tfidf(tfidf, text_input):
    tf1_new = TfidfVectorizer(stop_words=my_stop_words, sublinear_tf=True, min_df= 0, ngram_range=(1, 2), vocabulary = tfidf.vocabulary_)
    features = tf1_new.fit_transform(text_input)
    return features

def score(model_type, text_input):
     model=np.load('models/' + model_name+ '.pkl')
     tfidf=np.load('tfidf_vec.pkl')
     features = fit_tfidf(tfidf, text_input)
     prediction = model.predict(features)
     return prediction[0]
