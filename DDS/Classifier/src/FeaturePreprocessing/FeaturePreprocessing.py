# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from nltk import sent_tokenize

from sklearn.base import BaseEstimator, TransformerMixin

from FeaturePreprocessing.NltkUtils import NltkUtils


class FeaturePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.nltk_utils = NltkUtils()

    def fit(self, X, y=None):
        return self

    @staticmethod
    def inverse_transform(X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [list(self.tokenize(doc)) for doc in X]

    def preprocess_n_grams(self, word_tuples_list=[()]):
        new_word_list = []
        for tuples in word_tuples_list:
            word_list = []
            for word in tuples:
                token = self.nltk_utils.preprocess_token(token=word)
                word_list.append(token)
            new_word_list.append(tuple(word_list))
        return new_word_list

    def pos_tagger(self, sentence, n, stopwords=None, preprocess=True):

        sentence_list = self.nltk_utils.split_sentence(split_regex=r'\W+', sentence=sentence)
        n_grams = self.nltk_utils.create_ngrams(sentence_list=sentence_list, n=n)
        if preprocess:
            n_grams = self.preprocess_n_grams(word_tuples_list=n_grams)
        if stopwords:
            n_grams = self.nltk_utils.remove_stopwords(word_tuples_list=n_grams, stopwords=stopwords)
        joined_n_grams = self.nltk_utils.join_by(datas=n_grams, join_param=" ")

        return self.nltk_utils.base_pos_tag(list(filter(None, joined_n_grams)))

    def n_pos_tagger(self, sentence, n_grams=3, stopwords=None, preprocess=False):
        n_pos_tag = []
        for n in range(n_grams):
            n_pos_tag.extend(self.pos_tagger(sentence, n + 1, stopwords=stopwords, preprocess=preprocess))
        return n_pos_tag

    def tokenize(self, document):

        # Break the document into sentences
        sentences = sent_tokenize(document)

        for sentence in sentences:

            # Break the sentence into part of speech tagged tokens
            tokens_and_pos_tags = self.n_pos_tagger(sentence, n_grams=3,
                                                    preprocess=True,
                                                    stopwords=self.stopwords)

            for token, tag in tokens_and_pos_tags:

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


if __name__ == "__main__":
    feature_preprocessor = FeaturePreprocessor()
    document = "Harshawaradhan Jadhav, a Shiv Sena legislator from Kannad assembly constituency in Aurangabad district, sent his resignation to the state assembly speaker Haribhau Bagade and demanded that the state should come up with an ordinance and a bill in the assembly on Maratha reservation."
    feature_preprocessor.tokenize(document=document)