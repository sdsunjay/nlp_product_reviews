import numpy as np
import pandas as pd
import torch
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import os.path
import random
import re
from nltk.corpus import stopwords
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier

from pytorch_pretrained_bert import BertTokenizer

import pickle
import string

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

MAX_LEN = 512

def save_model(model, filename):
    print('Saving model to ' + filename)
    # filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # load the model from disk
    return pickle.load(open(filename, 'rb'))

def train(features, labels):
    print('starting training')
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    # Logistic Regressions
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)
    lr_clf.score(test_features, test_labels)
    print('Finished training logistic regression model 1')
    print("Logistic Regression Classifier 1 accuracy percent:",
            (nltk.classify.accuracy(lr_clf, test_features)) * 100)
    save_model(lr_clf, 'LR_classifier')

    # Logistic Regression with Sklearn and random state
    LG_classifier = SklearnClassifier(LogisticRegression(random_state=42))
    LG_classifier.train(train_features)
    print('Finished training logistic regression model 2')
    save_model(LG_classifier, 'LG_classifier')
    print("Logistic Regression Classifier 2 accuracy percent:",
            (nltk.classify.accuracy(LG_classifier, test_features)) * 100)

    # Naive Bayes Classifier
    NB_classifier = nltk.NaiveBayesClassifier.train(train_features)
    print("Classic Naive Bayes Classifier accuracy percent:",(nltk.classify.accuracy(NB_classifier, test_features)) * 100)
    save_model(NB_classifier, 'NB_classifier')

    SVC_classifier = SklearnClassifier(SVC(),sparse=False).train(train_features)
    SVC_classifier.train(train_features)
    print("C-Support Vector Classifier accuracy
            percent:",(nltk.classify.accuracy(SVC_classifier, test_features)) * 100)
    save_model(NB_classifier, 'SVC_classifier')

    LinearSVC_classifier1 = SklearnClassifier(SVC(kernel='linear', probability=True, tol=1e-3))
    LinearSVC_classifier1.train(train_features)
    print("Linear Support Vector Classifier 1 accuracy percent:",
            (nltk.classify.accuracy(LinearSVC_classifier1, test_features)) * 100)
    LinearSVC_classifier2 = SklearnClassifier(LinearSVC("l1", dual=False, tol=1e-3))
    LinearSVC_classifier2.train(train_features)
    print("Linear Support Vector Classifier 2 accuracy percent:",
            (nltk.classify.accuracy(LinearSVC_classifier2, test_features)) * 100)
    LinearSVC_classifier3 = SklearnClassifier(LinearSVC("l2", dual=False, tol=1e-3))
    LinearSVC_classifier3.train(train_features)
    print("Linear Support Vector Classifier 3 accuracy percent:",
            (nltk.classify.accuracy(LinearSVC_classifier3, test_features)) * 100)

def create_tensor(padded, model):
    print('create_tensor')
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

def tokenize(df, text_column_name):
    print('Startinh to tokenize ' + text_column_name)
    model, tokenizer, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model, tokenizer, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pretrained model/tokenizer
    try:
        tokenizer = tokenizer.from_pretrained(pretrained_weights)
        # tokenized = tokenizer.tokenize(df[text_column_name])
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        # model.resize_token_embeddings(len(tokenizer))
        model = model.from_pretrained(pretrained_weights)
	    ### Now let's save our model and tokenizer to a directory
        model.save_pretrained('./my_model/')
        tokenizer.save_pretrained('./my_model/')
    except:
        print('Tokenizer failed')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized = tokenizer.tokenize(df[text_column_name])
        # model.resize_token_embeddings(len(tokenizer))

    print('Finished tokenizing text')
    return (tokenized,model)

def truncate(text):
    """Truncate the text."""
    text = (text[:511]) if len(text) > MAX_LEN else text
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
            # print('LINE: ' + line)
            words = line.split()
            for word in words:
                word = word.lower()
                # print('WORD: ' + word)
                if word not in stop_words:
                    all_words[word] = True
        temp = all_words.keys()
        # removePunctuationFromList(temp)


        all_words1 = nltk.FreqDist(temp)
        print("All Words list length : ", len(all_words1))
        print(str(list(all_words1.keys())[:100]))

        return list(all_words.keys())[:20000]
        # use top 6000 words
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

def addWordsIn(text, all_words):
    length_flag = len(text) > MAX_LEN
    # if length_flag:
    #    print('line: ' + text)
    words = text.split()
    final_string = ""
    flag = False
    try:
        for word in words:
            word = word.lower()

            if word in all_words:
                if length_flag:
                    if len(word) > 4:
                        final_string += word
                        final_string += ' '
                        flag = True
                else:
                    final_string += word
                    final_string += ' '
                    flag = True
            else:
                flag = False
        if(flag):
            final_string = final_string[:-1]
    except Exception as e:
        print("Error")
        exit()
        # type error: " + str(e))
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
    print('adding words in all_words')
    df['clean_text'] = [addWordsIn(line, all_words) for line in df['clean_text']]

    df["clean_text"] = df['clean_text'].apply(truncate)
    # df.text = df.text.apply(lambda x: x.lower())
    # df.text = df.text.apply(lambda x: x.translate(None, string.punctuation))
    # df.clean_text = df.clean_text.apply(lambda x: x.translate(string.digits))
    # all_words = df.clean_text.apply(get_counts)
    # df["clean_text"] = df['text'].str.replace('[^\w\s]','')
    print('Finished reading and cleaning data')
    # print(df.head(30))
    return df

def main(training_filepath):
    """Main function of the program."""
    df = read_data(training_filepath)
    df.clean_text.to_csv('clean_text.csv')
    # split into training, validation, and test sets
    training, validation, test = np.array_split(df.head(900), 3)
    tokens,model = tokenize(training, 'clean_text')
    padded = padding(tokens)
    features = create_tensor(padded, model)
    labels = training['human_tag']
    train(features, labels)


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

