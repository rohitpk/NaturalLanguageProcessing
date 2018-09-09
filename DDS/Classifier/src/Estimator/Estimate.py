
# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import pickle
import sys
import os



MODEL_FILE_NAME = 'model_MultinomialNB_2018_08_01_1.5.pickle'
sys.path.append(os.path.join(os.path.dirname( __file__ ), '..'))
MODEL_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', MODEL_FILE_NAME))


# CURRENT_FOLDER = os.path.dirname(__file__)
# os.chdir("..")
# MODEL_FILE_PATH = os.path.join(CURRENT_FOLDER,MODEL_FILE_NAME)


def arg_pass(arg):
    return arg

def load_model(PATH=MODEL_FILE_PATH):
    with open(PATH, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model=None, data=[]):

    if not model:
        model = load_model()
    if data:
        y_pred = model.predict(data)
    else:
        data = [
            "cricket match got banned due to heavy rain in pune",
            "terrorist attack in chennai",
            "landslide in ladakh"
        ]
        y_pred = model.predict(data)

    predicted_out = model.labels_.inverse_transform(y_pred)
    for i in range(len(data)):
        print("({}, {})".format(data[i], predicted_out[i]))
    return predicted_out


def predict_(model=None, data=[]):

    if not model:
        model = load_model()
    if data:
        y_pred = model.predict(data)
    else:
        data = [
            "cricket match got banned due to heavy rain in pune",
            "terrorist attack in chennai",
            "landslide in ladakh"
        ]
        y_pred = model.predict(data)

    predicted_out = model.labels_.inverse_transform(y_pred)
    predicted_proba = model.predict_proba(data)
    predicted_log_proba = model.predict_log_proba(data)
    predicted_score = model.score(data, y_pred)

    list_of_json = []
    for i in range(len(data)):
        class_name = "Relevant to bikers" if str(predicted_out[i]) == "1" else "Irrelevant to bikers"
        data = {}
        data.update({"probability": max(predicted_proba[i])})
        data.update({"probability_class": class_name})
        data.update({"log_probability": max(predicted_log_proba[i])})
        data.update({"score": predicted_score})
        list_of_json.append(data)
    return list_of_json


if __name__=='__main__':
    predict_()