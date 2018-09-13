# !usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 13:35:30 2018
@author: Rohit Kewalramani
"""


import json
import time
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from FeaturePreprocessing.FeaturePreprocessing import FeaturePreprocessor
from FeatureEngineering.FeatureBoosting import FeatureBooster

from DataModelling import logger


log = logger(name="DataModelling.BuildModel")

DEFAULT_FILENAME_FOR_NEGATIVE_WORDS = "Data/negative_word_corpus.csv"
DEFAULT_COLUMN_NAME_FOR_NEGATIVE_WORDS = "negative_words"


def timeit(func):
    """
    Simple timing decorator
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        delta = time.time() - start
        return result, delta
    return wrapper


class BuildModel(object):

    def __init__(self):
        pass

    @staticmethod
    def preprocessor():
        return FeaturePreprocessor()

    @staticmethod
    def arg_pass(arg):
        return arg

    def vectorizer(self):
        return TfidfVectorizer(lowercase=False,
                               use_idf=True, tokenizer=self.arg_pass)

    @staticmethod
    def classifier():
        return SGDClassifier()


    def feature_booster(self):
        return FeatureBooster(vectorizer=self.vectorizer(), feature_boosting_scalar=1.5,
                              feature_file_name=DEFAULT_FILENAME_FOR_NEGATIVE_WORDS,
                              feature_column_name=DEFAULT_COLUMN_NAME_FOR_NEGATIVE_WORDS)

    def pipeline(self, preprocessor=None, classifier=None, feature_booster=None):

        if isinstance(preprocessor, object) and preprocessor is not None:
            preprocessor = preprocessor()
        else:
            preprocessor = self.preprocessor()

        if isinstance(classifier, object) and classifier is not None:
            classifier = classifier
        else:
            classifier = self.classifier()

        if isinstance(feature_booster, object) and feature_booster is not None:
            feature_booster = feature_booster
        else:
            feature_booster = self.feature_booster()

        model = Pipeline(
            [
                ('preprocessor', preprocessor),
                ('feature_booster', feature_booster),
                ('classifier', classifier),
            ]
        )
        return model

    @timeit
    def build(self, x, y=None, classifier=None, preprocessor=None, feature_booster=None):
        """
        Build function that builds a single model.
        """

        model = self.pipeline(
            preprocessor=preprocessor,
            classifier=classifier,
            feature_booster=feature_booster
        )

        model.fit(x, y)
        return model

    @timeit
    def build_and_evaluate(self, x, y, preprocessor=None, classifier=None,
                           feature_booster=None, outpath=None, verbose=True,
                           test_size=0.2, print_report=True, Kfold_test=True, n_splits=5):

        # Label encode the targets
        labels = LabelEncoder()
        y = labels.fit_transform(y)

        # Begin evaluation
        if verbose:
            log.info("Building for evaluation")

        if Kfold_test:
            log.info("Performing K-fold test...")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=0)
            split_count = 1
            average_accuracy = 0 ; accuracy_list = []
            average_precision = 0
            average_recall = 0
            avearage_f1_score = 0

            for train_index, test_index in skf.split(x, y):
                # print("TRAIN:", train_index, "TEST:", test_index)
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model, secs = self.build(
                    x=x_train,
                    y=y_train,
                    preprocessor=preprocessor,
                    classifier=classifier,
                    feature_booster=feature_booster
                )

                if verbose:
                    log.info("Evaluation model fit in {:0.3f} seconds".format(secs))
                    log.info("Classification Report for fold-{}:\n".format(split_count))

                y_pred = model.predict(x_test)

                report = clsr(y_test, y_pred, target_names=labels.classes_)
                accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
                f1_s = f1_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
                precision = precision_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
                recall = recall_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
                confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

                fpr, tpr, _ = roc_curve(y_test, y_pred)
                area_under_curve = auc(fpr, tpr)
                print ("Area under curve:", area_under_curve)

                accuracy_list.append(accuracy)

                # plt.plot(fpr, tpr)

                # plt.step(recall_, precision_, color='b', alpha=0.2, where='post')
                # plt.fill_between(recall_, precision_, step='post', alpha=0.2, color='b')

                # plt.xlabel('Recall')
                # plt.ylabel('Precision')
                # plt.show()

                if print_report:
                    print(report)
                    print("Accuracy: {}".format(accuracy))
                    print("F1-Score: {}".format(f1_s))
                    print("Precision: {}".format(precision))
                    print("Recall: {}".format(recall))
                    print("Confusion-Matrix: ")
                    print(confusion_mat)
                average_accuracy = (average_accuracy * (split_count - 1) + (accuracy)) / (split_count)
                avearage_f1_score = (avearage_f1_score * (split_count - 1) + (f1_s)) / (split_count)
                average_precision = (average_precision * (split_count - 1) + (precision)) / (split_count)
                average_recall = (average_recall * (split_count - 1) + (recall)) / (split_count)

                split_count += 1
            std_dev_accuracy = np.std(accuracy_list)
            if print_report:
                print("\n************ K-fold results ************")
                print("Average-Accuracy: {}".format(average_accuracy))
                print("Average-F1-Score: {}".format(avearage_f1_score))
                print("Average-Precision: {}".format(average_precision))
                print("Average-Recall: {}".format(average_recall))
                print("*******************************************")

        else:
            x_train, x_test, y_train, y_test = tts(x, y, test_size=test_size)

            if verbose:
                log.info("Completed train test split")

            model, secs = self.build(
                x=x_train,
                y=y_train,
                preprocessor=preprocessor,
                classifier=classifier,
                feature_booster=feature_booster
            )

            if verbose:
                log.info("Evaluation model fit in {:0.3f} seconds".format(secs))
                log.info("Classification Report:\n")

            y_pred = model.predict(x_test)

            report = clsr(y_test, y_pred, target_names=labels.classes_)
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
            f1_s = f1_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
            precision = precision_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
            recall = recall_score(y_true=y_test, y_pred=y_pred, labels=labels.classes_, average="weighted")
            confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

            # fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            area_under_curve = auc(fpr, tpr)
            print(area_under_curve)
            # plt.step(recall_, precision_, color='b', alpha=0.2, where='post')
            # plt.fill_between(recall_, precision_, step='post', alpha=0.2, color='b')

            if print_report:
                print(report)
                print("Accuracy: {}".format(accuracy))
                print("F1-Score: {}".format(f1_s))
                print("Precision: {}".format(precision))
                print("Recall: {}".format(recall))
                print("Confusion-Matrix: ")
                print(confusion_mat)

        if verbose:
            log.info("Building complete model and saving ...")

        model, secs = self.build(
            x=x,
            y=y,
            preprocessor=preprocessor,
            classifier=classifier,
            feature_booster=feature_booster
        )

        model.labels_ = labels

        if verbose:
            log.info("Complete model fit in {:0.3f} seconds".format(secs))

        if outpath:
            with open(outpath, 'wb') as doc:
                pickle.dump(model, doc)
            log.info("Model written out to {}".format(outpath))


        json_path = outpath.split(".")[0] + ".json"
        # import ipdb; ipdb.set_trace()
        with open(json_path, 'w') as outfile:
            model_json = model.named_steps["classifier"].get_params()
            model_json.update({"model_name": str(type(model.named_steps["classifier"]))})
            json.dump(model_json, outfile)

        return model, average_accuracy, std_dev_accuracy
