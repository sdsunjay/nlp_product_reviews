import pickle
import string

import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
# Training the model and Testing Accuracy on Validation data
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import os.path
import sys, traceback
import random
import re
from nltk.corpus import stopwords
import nltk.classify.util
import nltk.metrics

from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import string

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',')

    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df

def main():

    testing_filepath = 'data/clean_testing.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    df = read_data(testing_filepath)

    # filepaths = [ 'XGBoost_Classifier' ]
    filepaths = [ 'logistic_regression_classifier' ]

    # Check whether the specified
    # path exists or not
    for filepath in filepaths:
        output_path = 'output_' + filepath + '.csv'
        isExist = os.path.exists(filepath)
        if(isExist):
            print('Reading model from ' + filepath)
        else:
            print('Model not found in the app path (' + filepath +')')
            exit()
        model = load_model(filepath)
        print('Model: ' + filepath)
        y_test_pred = model.predict(df["clean_text"])
        df2["human_tag"] = y_test_pred
        header = ["ID", "human_tag"]
        print('Output: ' + output_path)
        df2.to_csv(output_path, columns = header)

if __name__ == "__main__":
    main()
