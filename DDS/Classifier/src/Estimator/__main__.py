
# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import warnings; warnings.simplefilter(action="ignore")

from Estimator.Estimate import load_model, predict

PATH = "model.pickle"

def arg_pass(arg):
    return arg

def main():
    model = load_model(PATH)
    predict(model=model, data=[])
    return


if __name__ == "__main__":
    predict()