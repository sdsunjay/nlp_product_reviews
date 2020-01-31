
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

MAX_TOKENS = 512
MAX_WORDS = 475

def save_model(model, filename):
    print('Saving model to ' + filename)
    # filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1):
    history = model.fit(X_train, y_train)

    print('Model: ' + model_name)
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, np.around(y_test_pred)))
    print(roc_auc_score(y_test, y_test_pred))
    score = history.score(X_test, y_test)
    print('Score: ' + str(score))
    # make predictions for test data
    predictions = [round(value) for value in y_test_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    y_final_pred = model.predict(testing_features)
    df1["human_tag"] = y_final_pred
    header = ["ID", "human_tag"]
    output_path = 'result/' + model_name + '.csv'
    print('Output: ' + output_path)
    df1.to_csv(output_path, columns = header)
    # save_model(model, model_name)


def trainClassifiers(features, labels, testing_features, df1):
    print('Starting training')
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

    # Create a simple Logistic Regression classifier
    # model_name = 'Logistic Regression (solver=lbfgs)'
    # model = LogisticRegression(max_iter=200)
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Logistic Regression classifier
    # model_name = 'Logistic Regression (penalty elasticnet)'
    # model = LogisticRegression(max_iter=200, solver='saga')
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Linear SVC classifier
    # model_name = 'Linear SVC (max_iter=5000)'
    # model = LinearSVC(random_state=42, max_iter=5000)
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # XGBoost
    model = XGBClassifier()
    model_name = 'XGB'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # TODO - Fix this and make it work
    # Naive Bayes Classifier
    # model = NaiveBayesClassifier()
    # model_name = 'Naive Bayes Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # GaussianNB Classifier
    # model = GaussianNB()
    # model_name = 'Gaussian Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # TODO - Fix this and make it work
    # MultinomialNB Classifier
    # model = MultinomialNB()
    # model_name = 'Multinomial Classifier'
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Multi-Layer Perceptron Classifier (3 layers)
    model = MLPClassifier(hidden_layer_sizes=(30,30,30))
    model_name = 'MLP3'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    model = MLPClassifier(hidden_layer_sizes=(50,50,50,50,50))
    # model_name = 'Multi-Layer Perceptron Classifier (5 layers)'
    model_name = 'MLP5'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    model = MLPClassifier(hidden_layer_sizes=(20,20,20,20,20,20,20,20,20,20))
    model_name = 'MLP10'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)
    
    model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100))
    model_name = 'MLP100'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Sparse Support Vector Classifier
    model = SklearnClassifier(SVC(),sparse=False)
    model_name = 'Sparse_SVC'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Linear Support Vector Classifier
    model_name = 'Linear_SVC' 
    model = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # l1 Support Vector Classifier
    model_name = 'Linear_SVC_l1' 
    model = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # l2 Support Vector Classifier
    model_name = 'Linear_SVC_l2' 
    model = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Train SGD with hinge penalty
    # model_name = "Stochastic Gradient Descent Classifier (hinge loss)"
    model_name = 'SGD_hinge_loss' 
    model = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5000, tol=None))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Train SGD with Elastic Net penalty
    model = SklearnClassifier(SGDClassifier(alpha=1e-3, random_state=42, penalty="elasticnet", max_iter=5000, tol=None))
    # model_name = "Stochastic Gradient Descent Classifier (elasticnet)"
    model_name = 'SGD_elasticnet' 
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Ridge Classifier
    model = SklearnClassifier(RidgeClassifier(alpha=0.5, tol=1e-2, solver="sag"))
    model_name = "Ridge"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Perceptron Classifier
    model = SklearnClassifier(Perceptron(max_iter=5000))
    model_name = "Perceptron"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

    # Passive-Aggressive Classifier
    model = SklearnClassifier(PassiveAggressiveClassifier(max_iter=1000))
    model_name = "Passive-Aggressive"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test, testing_features, df1)

def createTensor1(padded, model):
    print('create_tensor1')
    input_ids = torch.tensor(np.array(padded))
    with torch.no_grad():
        last_hidden_states = model(input_ids)
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:, 0, :].numpy()
        print('Finished creating features')
        return features

def padding(tokenized):
    print(len(tokenized))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
       

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # np.array(padded).shape
    print('Finished padding text')
    return padded

### Now let's save our model and tokenizer to a directory
# model.save_pretrained('./my_saved_model_directory/')
#tokenizer.save_pretrained('./my_saved_model_directory/')

def tokenizeText1(df, text_column_name, model_class, tokenizer_class,pretrained_weights):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize ' + text_column_name)
        # print(df.head(10))
        # (tokenizer_class, pretrained_weights) = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # want RoBERTa instead of distilBERT, Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        ## Want BERT instead of distilBERT? Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer,'bert-large-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        model.save_pretrained('./my_model/')
        tokenizer.save_pretrained('./my_model/')
        padded = padding(tokenized)
        return createTensor1(padded, model)
    except Exception:
        print("Exception in Tokenize code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        exit()
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenized = tokenizer.tokenize(df[text_column_name])
        # model.resize_token_embeddings(len(tokenizer))

    print('Finished tokenizing text')
    return (tokenized,model,tokenizer)

def tokenizeText2(df, text_column_name, model_class, tokenizer_class, pretrained_weights):
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize 2' + text_column_name)
        # tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokens = df[text_column_name].apply((lambda x: tokenizer.tokenize(x)[:511]))
        # print(tokens)
        tokenized = tokenizer.convert_tokens_to_ids(tokens)

        # tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
        padded = padding(tokenized[:511])
        return createTensor1(padded, model)
    except Exception:
        print("Exception in Tokenize code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        exit()
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenized = tokenizer.tokenize(df[text_column_name])
        # model.resize_token_embeddings(len(tokenizer))

    print('Finished tokenizing text')
    return (tokenized,model,tokenizer)

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',', index_col='ID')
    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df

def main():
    """Main function of the program."""
    # Specify path
    training_filepath = 'data/clean_training.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    df = read_data(training_filepath)
    # import pdb; pdb.set_trace()
    # df = df.sample(frac=0.5).reset_index(drop=True)
    df = df.dropna()
    
    # split into training, validation, and test sets
    training, test = np.array_split(df.head(16000), 2)
    labels = training['human_tag']   
    # print(df.clean_text.to_string(index=False))
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
   


    testing_filepath = 'data/clean_testing.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    df1 = read_data(testing_filepath)
    df1 = df1.dropna()
    testing_features  = tokenizeText1(df1, 'clean_text', model_class, tokenizer_class, pretrained_weights)


    trainClassifiers(features, labels, testing_features, df1)
    
    
    # features = tokenizeText2(df, 'clean_text', model_class)
    # features  = tokenizeText2(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)
    # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    # features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    # trainClassifiers(features, labels)

if __name__ == "__main__":
    main()
