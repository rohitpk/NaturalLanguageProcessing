
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
MODEL_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Data/Model', MODEL_FILE_NAME))


def load_model(path=MODEL_FILE_PATH):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


class Predict(object):

    def __init__(self, path=MODEL_FILE_PATH):
        self.model = load_model(path=path)

    def predict(self, data=None):
        out = {
            "probability": None,
            "probability_class": None,
            "log_probability": None,
            "score": None
        }

        if data:
            data = [data]
            y_pred = self.model.predict(data)
            predicted_out = self.model.labels_.inverse_transform(y_pred)
            predicted_proba = self.model.predict_proba(data)
            predicted_log_proba = self.model.predict_log_proba(data)
            predicted_score = self.model.score(data, y_pred)

            class_name = "Relevant to bikers" if str(predicted_out[0]) == "1" else "Irrelevant to bikers"

            out.update({"probability": max(predicted_proba[0])})
            out.update({"probability_class": class_name})
            out.update({"log_probability": max(predicted_log_proba[0])})
            out.update({"score": predicted_score})

        return out
