from api.ml.model_pipeline import train, predict, get_accuracy


train('LogisticRegression')
print(predict(['hello my name is Carly'],'LogisticRegression'))

