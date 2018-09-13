# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 13:35:30 2018
@author: Vijayasai
"""

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import numpy as np

import operator


class FeatureSelection(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X

    @staticmethod
    def selection(X, y=None, **fit_params):
        chi, p_val = chi2(X, y)

        x = np.arange(0, len(p_val))
        zipped = zip(feature_names, chi, p_val)
        sorted_zipped = sorted(zipped, key=operator.itemgetter(1))

        feature_names, chi, p_val = zip(*sorted_zipped)

        threshold_chi_value = 0.05
        all_index = []
        for index, val in enumerate(chi):
            if val > threshold_chi_value:
                all_index.append(index)
        print ("Total Features: ", len(all_index))

        new_X = X[:, all_index]

        plt.plot(x, chi, "r.")
        plt.show()
        plt.plot(x, p_val, "b.")
        plt.show()
        return new_X

    def fit_transform(self, X, y=None, **fit_parmas):
        self.fit(X, y, **fit_parmas)
        return self.transform(X, y, **fit_parmas)
