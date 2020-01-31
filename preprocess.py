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
from sklearn.neighbors.nearest_centroid import NearestCentroid
from xgboost import XGBClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from pytorch_pretrained_bert import BertTokenizer

import pickle
import string

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

MAX_TOKENS = 512
MAX_WORDS = 475


def truncate(text):
    """Truncate the text."""
    # TODO fix this to use a variable instead of 511
    text = (text[:511]) if len(text) > MAX_TOKENS else text
    return text

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def contractions(text):
    contractions = {
        "ain't": "are not ",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i had",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as ",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"}

    words = text.split()
    final_string = ""
    try:
        for word in words:
            word = word.lower()
            if hasNumbers(word) == False:
                if word in contractions:
                    # print('Word: ' + word)
                    # print('Replacement: ' + contractions[word])
                    final_string += contractions[word]
                    final_string += ' '
                    flag = True
                else:
                    final_string += word
                    final_string += ' '
                    flag = False
        if(flag):
            final_string = final_string[:-1]
    except Exception as e:
        print("type error: " + str(e))
        exit()
    return final_string

def removePunctuationFromList(all_words):
    all_words = [''.join(c for c in s if c not in string.punctuation)
            for s in all_words]
    # Remove the empty strings:
    all_words = [s for s in all_words if s]
    return all_words

def cleanText(text):
    """Clean up the text."""
    try:
        text = str(text)

        # remove contactions and stop words
        text = contractions(text)
        # remove html entities
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        new_text = cleanr.sub('', text.strip())
        return re.sub(r'\s+', ' ', re.sub(r'\W+', " ", new_text))
        # TAG_RE = re.compile(r'<[^>]+>')
    except:
        print("An exception occurred with: " + text)
        return str(text)

def getAllWords(lines, stop_words):
    all_words = {}
    try:
        for line in lines:
            words = line.split()
            for word in words:
                if word not in stop_words:
                    all_words[word] = True
        temp = all_words.keys()
        # removePunctuationFromList(temp)


        top_words = nltk.FreqDist(temp)
        print("All Words list length : ", len(top_words))
        # print(str(list(all_words1.keys())[:100]))

        # use top 20000 words
        return list(top_words.keys())[:20000]
        # word_features = list(all_words.keys())[:6000]
        # featuresets = [(find_features(rev, word_features), category)
        #        for (rev, category) in documents]
        # print("Feature sets list length : ", len(featuresets))
    except Exception as e:
        print("type error: " + str(e))
        exit()



def removeWordsNotIn(text, stop_words):
    words = text.split()
    final_string = ""
    flag = False
    try:
        for word in words:
            word = word.lower()
            if word not in stop_words:
                final_string += word
                final_string += ' '
                flag = True
            else:
                flag = False
        if(flag):
            final_string = final_string[:-1]
    except Exception as e:
        # print("type error: " + str(e))
        print("type error")
        exit()
    return final_string

def shortenText(text, all_words):
    # print('shortenText')
    count = 0
    final_string = ""
    try:
        words = text.split()
        for word in words:
            word = word.lower()
            if len(word) > 7:
                if word in all_words:
                    count += 1
                    if(count == MAX_WORDS-1):
                        # if we hit max number of token, stop parsing string
                        return final_string[:-1]
                    else:
                        final_string += word
                        final_string += ' '
        final_string = final_string[:-1]
    except Exception as e:
        print("Error")
        # exit()
        print("type error: " + str(e))
    return final_string


def addWordsIn(text, all_words):
    """ Also does truncation """
    count = 0
    final_string = ""
    try:
        words = text.split()
        for word in words:
            word = word.lower()

            if word in all_words:
                count += 1
                if(count == MAX_WORDS-1):
                    return shortenText(text, all_words)
                else:
                    final_string += word
                    final_string += ' '
        final_string = final_string[:-1]
    except Exception as e:
        print("Error")
        # exit()
        print("type error: " + str(e))
    return final_string

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',')

    stop_words = ["will", "done", "goes","let", "know", "just", "put" "also",
            "got", "can", "get" "said", "mr", "mrs", "one", "two", "three",
            "four", "five", "i", "me", "my", "myself", "we", "our",
            "ours","ourselves","you","youre","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how",
            "all","any","both","each","few","more","most","other","some","such",
            "can", "will",
            "just",
            "don",
            "don't",
            "should",
            "should've",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "aren't",
            "couldn",
            "couldn't",
            "didn",
            "didn't",
            "doesn",
            "doesn't",
            "hadn",
            "hadn't",
            "hasn",
            "hasn't",
            "haven",
            "haven't",
            "isn",
            "isn't",
            "ma",
            "mightn",
            "mightn't",
            "mustn",
            "mustn't",
            "needn",
            "needn't",
            "shan",
            "shan't",
            "shouldn"
            "shouldn't",
            "wasn",
            "wasn't",
            "weren",
            "weren't",
    "won",
"won't",
"wouldn",
"wouldn't"]

    # pandas drop columns using list of column names
    df = df.drop(['doc_id', 'date', 'title'], axis=1)
    print('Cleaning text')
    df["clean_text"] = df['text'].apply(cleanText)
    print('Removing words in stop words')
    df['clean_text'] = [removeWordsNotIn(line, stop_words) for line in df['clean_text']]

    clean_text = df["clean_text"].tolist()
    # print(clean_text[:10])
    print('Getting all words')
    all_words = getAllWords(clean_text, stop_words)
    # print('adding words in all_words')
    df['clean_text'] = [addWordsIn(line, all_words) for line in df['clean_text']]

    # df.text = df.text.apply(lambda x: x.translate(None, string.punctuation))
    # df.clean_text = df.clean_text.apply(lambda x: x.translate(string.digits))
    # df["clean_text"] = df['text'].str.replace('[^\w\s]','')
    print('Finished reading and cleaning data')
    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df

def preprocess_file(filepath, output_path, flag):

    df = read_data(filepath)
    if(flag):
        header = ["ID", "clean_text", "star_rating", 'human_tag']
    else:
        header = ["ID", "clean_text"]
    print('Output: ' + output_path)
    df.to_csv(output_path, columns = header, index=False)

def main():
    """Main function of the program."""
    # Specify path
    training_filepath = 'data/training.csv'
    testing_filepath = 'data/public_test_features.csv'

    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    preprocess_file(training_filepath, 'data/clean_training.csv', True)
    # Check whether the specified path exists or not
    isExist = os.path.exists(testing_filepath)
    if(isExist):
        print('Reading from ' + testing_filepath)
    else:
        print('Testing file not found in the app path.')
        exit()
    preprocess_file(testing_filepath,'data/clean_testing.csv', False)

if __name__ == "__main__":
    main()

