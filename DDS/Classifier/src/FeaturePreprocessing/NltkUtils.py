# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import re
from nltk import ngrams
from nltk import pos_tag


class NltkUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def create_ngrams(sentence_list=[], n=1):
        return list(ngrams(sentence_list, n))

    @staticmethod
    def split_sentence(split_regex=r'\W+', sentence=""):
        return re.split(split_regex, sentence)

    @staticmethod
    def base_pos_tag(words_list=[]):
        return pos_tag(words_list)

    @staticmethod
    def join_by(join_param=" ", datas=[]):
        return [join_param.join(data) for data in datas]

    @staticmethod
    def remove_stopwords(word_tuples_list=[()], stopwords=None):

        if stopwords:
            new_word_list = []
            for tuples in word_tuples_list:
                to_append = True
                for word in tuples:
                    if word in stopwords:
                        to_append = False
                if to_append:
                    new_word_list.append(tuples)
            return new_word_list
        else:
            return word_tuples_list

    @staticmethod
    def preprocess_token(token, lower=True, strip=True):

        # Apply preprocessing to the token
        token = token.lower() if lower else token
        token = token.strip() if strip else token
        token = token.strip('_') if strip else token
        token = token.strip('*') if strip else token
        return token
