from common import getDataFrame, tokenizeText1

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

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

from pytorch_pretrained_bert import BertTokenizer

import pickle
import string

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def main():
    """Main function of the program."""
    
    begin_time_main = datetime.datetime.now()
    print("Begin time: " + begin_time_main)
    df = getDataFrame()
    
    # split into training, validation, and test sets
    training, test = np.array_split(df.head(1000), 2)
    labels = training['human_tag']   
   
    train = torch.utils.data.TensorDataset(torch.Tensor(np.array(df)), torch.Tensor(np.array(labels)))
    train_loader = torch.utils.data.DataLoader(train, batch_size = 100, shuffle = True, num_workers=4)
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features = []
    for i, batch in enumerate(train_loader):   
    	# When we have more time
    	# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-large-uncased') 
    	features.append(tokenizeText1(batch, 'clean_text', model_class, tokenizer_class, pretrained_weights))

    testing_filepath = 'data/clean_testing1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    df1 = read_data(testing_filepath)
    df1 = df1.dropna()
    a = np.array_split(df1,8)
    i = 0
    values = []
    for aa in a: 
        output_name = str(i) 
        print('Run: ' + output_name)
        i += 1
        testing_features  = tokenizeText1(aa, 'clean_text', model_class, tokenizer_class, pretrained_weights)
        final_y_pred = trainClassifiers(features, labels, testing_features)
        values = np.concatenate((values, final_y_pred), axis=0)

    df1["human_tag"] = values
    header = ["ID", "human_tag"]
    output_path = 'result/MLP100' 
    print('Output: ' + output_path)
    df1.to_csv(output_path, columns = header)
    print(datetime.datetime.now() - begin_time_main)
    
    # features = tokenizeText2(df, 'clean_text', model_class)
    # features  = tokenizeText2(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)
    # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)

if __name__ == "__main__":
    main()
