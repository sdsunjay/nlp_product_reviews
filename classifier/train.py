
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

def save_model(model, filename):
    print('Saving model to ' + filename)
    # filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def trainClassifier(model_name, model, X_train, X_test, y_train, y_test):
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
    # save_model(model, model_name)


def trainClassifiers(features, labels):
    print('Starting training')
    # create training and testing vars
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Create a simple Logistic Regression classifier
    model_name = 'Logistic Regression (solver=lbfgs)'
    model = LogisticRegression(max_iter=200)
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Logistic Regression classifier
    model_name = 'Logistic Regression (penalty elasticnet)'
    model = LogisticRegression(max_iter=200, solver='saga', penalty='elasticnet')
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Create a simple Linear SVC classifier
    model_name = 'Linear SVC (max_iter=2000)'
    model = LinearSVC(random_state=42, max_iter=2000)
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # XGBoost
    model = XGBClassifier()
    model_name = 'XGBoost Classifier'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Naive Bayes Classifier
    model = NaiveBayesClassifier()
    model_name = 'Naive Bayes Classifier'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Sparse Support Vector Classifier
    model = SklearnClassifier(SVC(),sparse=False).train(train_features)
    model_name = 'Sparse Support Vector Classifier'
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Linear Support Vector Classifier
    model_name = 'Linear Support Vector Classifier 1'
    model = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # l1 Support Vector Classifier
    model_name = 'Linear Support Vector Classifier 2 (l1)'
    model = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # l2 Support Vector Classifier
    model_name = 'Linear Support Vector Classifier 3 (l2)'
    model = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Train SGD with hinge penalty
    model_name = "Stochastic Gradient Descent Classifier (hinge loss)"
    model = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5000, tol=None))
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Train SGD with Elastic Net penalty
    model = SklearnClassifier(SGDClassifier(alpha=1e-3, random_state=42, penalty="elasticnet", max_iter=5000, tol=None))
    model_name = "Stochastic Gradient Descent Classifier (elasticnet)"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Deprecated
    # Train NearestCentroid (aka Rocchio classifier) without threshold
    # model = SklearnClassifier(NearestCentroid())
    # model_name = "Nearest Centroid Classifier without threshold"
    # trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Ridge Classifier
    model = SklearnClassifier(RidgeClassifier(alpha=0.5, tol=1e-2, solver="sag"))
    model_name = "Ridge Classifier"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Perceptron Classifier
    model = SklearnClassifier(Perceptron(max_iter=5000))
    model_name = "Perceptron Classifier"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

    # Passive-Aggressive Classifier
    model = SklearnClassifier(PassiveAggressiveClassifier(max_iter=1000))
    model_name = "Passive-Aggressive Classifier"
    trainClassifier(model_name, model, X_train, X_test, y_train, y_test)

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
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # np.array(padded).shape
    print('Finished padding text')
    return padded

def something(df, text_column_name):
	### Let's load a model and tokenizer
	model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	### Do some stuff to our model and tokenizer
	# Ex: add new tokens to the vocabulary and embeddings of our model
	tokenizer.add_tokens(['[SPECIAL_TOKEN_1]', '[SPECIAL_TOKEN_2]'])
	model.resize_token_embeddings(len(tokenizer))
	# Train our model
	train(model)

	### Now let's save our model and tokenizer to a directory
	model.save_pretrained('./my_saved_model_directory/')
	tokenizer.save_pretrained('./my_saved_model_directory/')

def tokenizeText1(df, text_column_name, model_class, tokenizer_class,
        pretrained_weights):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize ' + text_column_name)
        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # want RoBERTa instead of distilBERT, Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        ## Want BERT instead of distilBERT? Uncomment the following line:
        # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer,'bert-large-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))
        # tokens = df[text_column_name].apply((lambda x: tokenizer.tokenize(x)[:511]))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
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

def tokenizeText2(df, text_column_name, model_class):
    # Load pretrained model/tokenizer
    try:
        print('Starting to tokenize 2' + text_column_name)
        tokenizer_class, pretrained_weights = (ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        model = model_class.from_pretrained(pretrained_weights)
        tokenizer = tokenizer_class.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        model.resize_token_embeddings(len(tokenizer))
        tokens = df[text_column_name].apply((lambda x: tokenizer.tokenize(x)))
        tokenized = tokenizer.convert_tokens_to_ids(tokens)
        # tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True)))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
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
        # source = text
        # source = source.str.replace('[^A-Za-z]',' ')
        # data['description'] = data['description'].str.replace('\W+',' ')
        # source = source.str.lower()
        # source = source.str.replace("\s\s+" , " ")
        # source = source.str.replace('\s+[a-z]{1,2}(?!\S)',' ')
        # source = source.str.replace("\s\s+" , " ")
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
    # print('Adding only the top words')
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
                    # if we hit max number of token, stop parsing string
                    # return final_string[:-1]
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
    df = df.drop(['ID', 'doc_id', 'date', 'title', 'star_rating'], axis=1)
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
    # print(df.head(30))
    return df

def main(training_filepath):
    """Main function of the program."""
    df = read_data(training_filepath)
    # df.clean_text.to_csv('clean_text.csv')
    # split into training, validation, and test sets
    training, test = np.array_split(df.head(1000), 2)
    labels = training['human_tag']

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    trainClassifiers(features, labels)
    model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    features  = tokenizeText1(training, 'clean_text', model_class, tokenizer_class, pretrained_weights)
    trainClassifiers(features, labels)

    # features  = tokenizeText2(training, 'clean_text',ppb.DistilBertModel)



if __name__ == "__main__":
    # Specify path
    training_filepath = 'data/training.csv'

    # Check whether the specified
    # path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    main(training_filepath)

