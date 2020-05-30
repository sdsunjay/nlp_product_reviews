import pandas as pd
import numpy as np

import os
import sys, traceback

from datetime import datetime

import torch
import transformers as ppb  # pytorch transformers
from myClass import Dataset

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',', index_col='ID')
    print('Number of rows in dataframe: ' + str(len(df.index)))
    return df


def getDataFrame():

    # Specify path
    training_filepath = 'data/clean_training1.csv'
    # Check whether the specified path exists or not
    isExist = os.path.exists(training_filepath)
    if(isExist):
        print('Reading from ' + training_filepath)
    else:
        print('Training file not found in the app path.')
        exit()
    df = read_data(training_filepath)
    # import pdb; pdb.set_trace()
    df = df.sample(frac=0.5).reset_index(drop=True)
    df = df.dropna()
    return df

def createTensor(padded, model):
    begin_time_main = datetime.now()
    print("Create Tensor begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        # last_hidden_states = model(input_ids)
        last_hidden_states = model(input_ids, attention_mask=attention_mask)    
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:,0,:].numpy()        
        print('Create Tensor end time: ' + str(datetime.now() - begin_time_main))
        return features

def padding(tokenized):
    begin_time_main = datetime.now()
    print("Padding begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    
    print(len(tokenized))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
       

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # np.array(padded).shape
    print('Padding end time: ' + str(datetime.now() - begin_time_main))
    return padded

def tokenizeText1(df, text_column_name, model_class, tokenizer_class, pretrained_weights):
    begin_time_main = datetime.now()
    print("tokenize begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
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
        tokenized = df[text_column_name].apply((lambda x: tokenizer.encode(x,add_special_tokens=True, max_length=511)))

        ### Now let's save our model and tokenizer to a directory
        # model.save_pretrained('./my_model/')
        # tokenizer.save_pretrained('./my_model/')
        padded = padding(tokenized)
        result = createTensor(padded, model)
        print('tokenize end time: ', str(datetime.now() - begin_time_main))
        return result
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
