# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import pandas as pd
from FeatureEngineering import logger
from sklearn.base import TransformerMixin, BaseEstimator

log = logger("FeatureEngineering.FeatureBoosting")

DEFAULT_FILENAME_FOR_NEGATIVE_WORDS = "Data/negative_word_corpus.csv"
DEFAULT_COLUMN_NAME_FOR_NEGATIVE_WORDS = "negative_words"


class FeatureBooster(TransformerMixin, BaseEstimator):

    def __init__(self, vectorizer=None, feature_boosting_scalar=2, feature_selector=None,
                 feature_file_name=DEFAULT_FILENAME_FOR_NEGATIVE_WORDS,
                 feature_column_name=DEFAULT_COLUMN_NAME_FOR_NEGATIVE_WORDS):

        df = pd.read_csv(feature_file_name)
        self.word_list = set(df[feature_column_name].tolist())

        self.vectorizer = vectorizer
        self.feature_selector = feature_selector
        self.feature_boosting_scalar = feature_boosting_scalar
        self.feature_names = None
        log.info("Feature boosting scalar multiplier value: {}".format(self.feature_boosting_scalar))

    def transform(self, X, y=None, **fit_params):

        X = self.vectorizer.transform(X)
        self.feature_names = self.vectorizer.get_feature_names()

        # new_X = self.feature_selector.transform(X, y, feature_names=self.feature_names)
        feature_index = set(X[:, :].nonzero()[1])

        for index in feature_index:
            feature_name = self.feature_names[index]
            if feature_name in self.word_list:
                X[:, index] *= self.feature_boosting_scalar
                # log.info("Feature {}'s DTM value is scaled up".format(feature_name))

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)

    def fit(self, X, y=None, **fit_params):
        self.vectorizer.fit(X)
        return self
