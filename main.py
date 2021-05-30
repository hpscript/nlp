# -*- coding: utf-8 -*-
#! /usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os.path
import nlp_tasks

from sklearn.neural_network import MLPClassifier # アルゴリズムのmlp

def train():
    classifier = MyMLPClassifier()
    classifier.train('corpus.csv')

def predict():
    classifier = MyMLPClassifier()
    classifier.load_model()
    result = classifier.predict(u"競馬の3歳馬日本一を決めるG1レース、「第88回日本ダービー」が行われ、4番人気のシャフリヤールが優勝しました。")
    print(result)



class MyMLPClassifier():
    model = None
    model_name = "mlp"

    def load_model(self):
    	if os.path.exists(self.get_model_path())==False:
        	raise Exception('no model file found!')
    	self.model = joblib.load(self.get_model_path())
    	self.classes =  joblib.load(self.get_model_path('class')).tolist()
    	self.vectorizer = joblib.load(self.get_model_path('vect'))
    	self.le = joblib.load(self.get_model_path('le'))

    def get_model_path(self,type='model'):
    	return 'models/'+self.model_name+"_"+type+'.pkl'

    def get_vector(self,text):
    	return self.vectorizer.transform([text])



    def train(self, csvfile):
    	df = pd.read_csv(csvfile,names=('text','category'))
    	X, vectorizer = nlp_tasks.get_vector_by_text_list(df["text"])

    	le = LabelEncoder()
    	le.fit(df['category'])
    	Y = le.transform(df['category'])

    	model = MLPClassifier(max_iter=300, hidden_layer_sizes=(100,),verbose=10,)
    	model.fit(X, Y)

    	joblib.dump(model, self.get_model_path())
    	joblib.dump(le.classes_, self.get_model_path("class"))
    	joblib.dump(vectorizer, self.get_model_path("vect"))
    	joblib.dump(le, self.get_model_path("le"))

    	self.model = model
    	self.classes = le.classes_.tolist()
    	self.vectorizer = vectorizer

    def predict(self,query):
    	X = self.vectorizer.transform([query])
    	key = self.model.predict(X)
    	return self.classes[key[0]]


if __name__ == '__main__':
    # train()
    predict()