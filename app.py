from flask import Flask,request, url_for, redirect, render_template
import numpy as np
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

prediction_labels = {'Emotional pain': 0, 'Hair falling out':1, 'Head hurts':2, 'Infected wound':3, 'Foot achne':4,
    'Shoulder pain':5, 'Injury from sports':6, 'Skin issue':7, 'Stomach ache':8, 'Knee pain':9, 'Joint pain':10, 'Hard to breath':11,
    'Head ache':12, 'Body feels weak':13, 'Feeling dizzy':14, 'Back pain':15, 'Open wound':16, 'Internal pain':17, 'Blurry vision':18,
    'Acne':19, 'Neck pain':21, 'Cough':22, 'Ear achne':23, 'Feeling cold':24}

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

model = load_model('model.pkl')

vec_model = load_model('vectorizer.pkl')


app = Flask(__name__)



@app.route('/')

def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    vec_text =  vec_model.transform([x for x in request.form.values()]).toarray()
    pred = model.predict(vec_text)
    final_result = get_key(pred,prediction_labels)
    print(final_result)
    
    return render_template('index.html',pred='According to our estimate you are suffering from {}'.format(final_result))

if __name__ == '__main__':
    app.run(debug=True)