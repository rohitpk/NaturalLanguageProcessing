# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:35:30 2018
@author: Rohit Kewalramani
"""

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
import pandas as pd

import pyLDAvis.sklearn
# pyLDAvis.enable_notebook()

def get_data():
    data = []

    for fileid in brown.fileids():
        document = ' '.join(brown.words(fileid))
        data.append(document)
    return data


def get_data_from_csv(csv_name='news_training_set.csv'):
    df = pd.read_csv(csv_name)
    data = df['title_content'].tolist()
    return data



def vectorize_data(data):

    vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                                 stop_words='english', lowercase=True,
                                 token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
    data_vectorized = vectorizer.fit_transform(data)

    return vectorizer,data_vectorized


def build_lda_model(vectorized_data,number_of_topics=10):
    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(n_topics=number_of_topics, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(vectorized_data)
    # print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    return lda_Z,lda_model

def build_nmf_model(vectorized_data,number_of_topics=10):
    # Build a Non-Negative Matrix Factorization Model
    nmf_model = NMF(n_components=number_of_topics)
    nmf_Z = nmf_model.fit_transform(vectorized_data)
    # print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    return nmf_Z,nmf_model


def build_lsa_model(vectorized_data,number_of_topics=10):
    # Build a Latent Semantic Indexing Model
    lsa_model = TruncatedSVD(n_components=number_of_topics)
    lsa_Z = lsa_model.fit_transform(vectorized_data)
    # print(lsa_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    return lsa_Z,lsa_model


def print_topics(model, vectorizer, top_n=10):
    print(dir(model))
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])




def main():
    data = get_data_from_csv()
    vectorizer,vectorized_data = vectorize_data(data)

    lda_Z,lda_model = build_lda_model(vectorized_data,10)
    lsa_Z,lsa_model = build_lsa_model(vectorized_data,10)
    nmf_Z,nmf_model = build_nmf_model(vectorized_data,10)
# Let's see how the first document in the corpus looks like in different topic spaces
#     print(lda_Z[0])
    # print(lsa_Z[0])
    # print(nmf_Z[0])

    print("LDA Model:")
    print_topics(lda_model, vectorizer)
    print("=" * 20)
    #
    print("NMF Model:")
    print_topics(nmf_model, vectorizer)
    print("=" * 20)

    print("LSA Model:")
    print_topics(lsa_model, vectorizer)
    print("=" * 20)

    # panel = pyLDAvis.sklearn.prepare(lda_model, vectorized_data, vectorizer, mds='tsne')
    panel = pyLDAvis.sklearn.prepare(nmf_model, vectorized_data, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel,'nmf_op.html')
    # print(dir(panel))

if __name__=='__main__':
    main()
#
#
# text = "The economy is working better than ever"
# x = nmf_model.transform(vectorizer.transform([text]))[0]
# print(x)
#
# from sklearn.metrics.pairwise import euclidean_distances
#
# def most_similar(x, Z, top_n=5):
#     dists = euclidean_distances(x.reshape(1, -1), Z)
#     pairs = enumerate(dists[0])
#     most_similar = sorted(pairs, key=lambda item: item[1])[:top_n]
#     return most_similar
#
# similarities = most_similar(x, nmf_Z)
# document_id, similarity = similarities[0]
# print(data[document_id][:1000])