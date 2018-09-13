# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Vijayasai
"""

import warnings; warnings.simplefilter(action="ignore")
import pandas as pd
import numpy as np

df = pd.read_csv("Data/temp/refined/training_data.csv")


def get_x():
    return np.array(df["headline_text"].tolist())


def get_y():
    return np.array(list(map(str, df["class_tag"].tolist())))


def training_data():
    x = get_x()
    y = get_y()
    return x, y
