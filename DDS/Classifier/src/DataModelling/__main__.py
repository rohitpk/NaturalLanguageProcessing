# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""

import warnings; warnings.simplefilter("ignore")
import argparse

from TrainingData.FetchDataCsv import training_data

from DataModelling.BuildModel import BuildModel
from DataModelling import logger

# Classification Model
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# Vectorization Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Feature Boosting Model
from FeatureEngineering.FeatureBoosting import FeatureBooster
# from FeatureSelection.FeatureSelection import FeatureSelection

build_model = BuildModel()
log = logger(name="DataModelling.__main__")

parser = argparse.ArgumentParser(description="DataModelling")
parser.add_argument("--boosterMultiplier", type=float,
                    default=1.0, help="Multiplier to boost the features in DTM Matrix")
parser.add_argument("--testSize", type=float,
                    default=0.2, help="Test_size fraction for train test split")
parser.add_argument("--verbose", type=bool,
                    default=True, help="Print's the logs on the terminal")
parser.add_argument("--printReport", type=bool,
                    default=True, help="Print's the classification report")
parser.add_argument("--vectorizer", type=str,
                    default="TfidfVectorizer", help="Vectorizer for the Document-Term "
                                                    "matrix (DTM). Vectorizer available are "
                                                    "TfidfVectorizer, CountVectorizer. "
                                                    "Default TfidfVectorizer.")
parser.add_argument("--fileToSave", type=str,
                    default="model.pickle", help="Filename or pathname to store "
                                                 "the model")
parser.add_argument("--classifier", type=str,
                    default="SGDClassifier", help="Set the classifier for "
                                                  "building the model. Classifiers "
                                                  "available are SGDClassifier, SVC, "
                                                  "MultinomialNB, LogisticRegression "
                                                  "GradientBoostingClassifier, MLPClassifier, "
                                                  "AdaBoostClassifier, RandomForestClassifier. "
                                                  "Default SGDClassifier.")

args = parser.parse_args()


def arg_pass(arg):
    return arg


def main():
    vectorizer = args.vectorizer
    classifier = args.classifier

    classifier = classifier.split(",")

    fileToSave = args.fileToSave
    verbose = args.verbose
    printReport = args.printReport
    test_size = args.testSize
    feature_boosting_scaler = args.boosterMultiplier

    log.info("Fetching data for training")
    x, y = training_data()
    log.info("Started modelling the data")

    classifiers = dict(
        MultinomialNB=MultinomialNB(),
        SVC=SVC(),
        SGDClassifier=SGDClassifier(),
        RandomForestClassifier=RandomForestClassifier(),
        AdaBoostClassifier=AdaBoostClassifier(),
        MLPClassifier=MLPClassifier(),
        LogisticRegression=LogisticRegression(),
        GradientBoostingClassifier=GradientBoostingClassifier()
    )

    log.info("Classifier used for modelling: ")
    print (classifier)

    if vectorizer == "TfidfVectorizer":
        vectorizer = TfidfVectorizer(lowercase=False, use_idf=True, tokenizer=arg_pass)

    elif vectorizer == "CountVectorizer":
        vectorizer = CountVectorizer(lowercase=False, tokenizer=arg_pass)

    else:
        log.debug("No vectorizer name called {} is available for modelling.".format(vectorizer))
        return
    log.info("Vectorizer used for modelling: ")
    print (vectorizer)

    log.info("Train Data: {}%, Test Data: {}%".format((1 - test_size)*100, test_size*100))

    feature_booster = FeatureBooster(vectorizer=vectorizer,
                                     feature_boosting_scalar=feature_boosting_scaler)

    log.info("Multiplier to boost the features in DTM (CSR) Matrix")

    average_accuracy_model = []
    std_accuracy_model = []
    for c in classifier:
        print ('-'*45,'\n', c)
        c = classifiers.get(c, None)
        if c:
            model, secs = build_model.build_and_evaluate(
                                x=x,
                                y=y,
                                outpath=fileToSave,
                                classifier=c,
                                feature_booster=feature_booster,
                                verbose=verbose,
                                test_size=test_size,
                                print_report=printReport,
                                Kfold_test=True
                            )
            # plt.plot(model[2], model[1])
            average_accuracy_model.append(model[1])
            std_accuracy_model.append(model[2])
            log.info("Building and Evaluation model fit in {:0.3f} seconds".format(secs))
        else:
            log.debug("No classifier name called {} is available for modelling.".format(c))

    # plt.bar([1,2,3,4], height=average_accuracy_model)
    # plt.xticks([1,2,3,4], classifier)
    # plt.ylabel("Mean-Accuracy")
    # plt.title("5-fold validation of classification models")
    # plt.show()
    # plt.bar([1,2,3,4], height=std_accuracy_model)
    # plt.xticks([1,2,3,4], classifier)
    # plt.ylabel("StdDev-Accuracy")
    # plt.title("5-fold validation of classification models")
    # plt.show()
    log.info("Completed modelling the data")
    return


if __name__ == "__main__":
    main()
