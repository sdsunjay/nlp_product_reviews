import pandas as pd
import numpy as np

import os
import sys, traceback

from datetime import datetime

import torch
import transformers as ppb  # pytorch transformers

def read_data(filepath):
    """Read the CSV from disk."""
    df = pd.read_csv(filepath, delimiter=',', index_col='ID')
    # df = pd.read_csv(filepath, delimiter=',', skiprows = 1)
    print('Number of rows in dataframe: ' + str(len(df.index)))
    print(df.head(5))
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

def batch(padded, labels, model):
    begin_time_main = datetime.now()
    print("batch begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    # Parameters
    params = {'batch_size': 200,
              'shuffle': False,
              'num_workers': 8}
    max_epochs = 100

    # Datasets
    # partition = # IDs
    # labels = # Labels

    # Generators
    # training_set = Dataset(padded, labels)
    training_data = torch.utils.data.DataLoader(padded, **params)

    # validation_set = Dataset(partition['validation'], labels)
    # validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    final_input_ids = torch.zeros(0.0,0.0)
    # Loop over epochs
    # for epoch in range(max_epochs):

    # Training
    for local_batch in training_data:
        input_ids = torch.tensor(local_batch)
        attention_mask = np.where(local_batch != 0, 1, 0)
        attention_mask = torch.tensor(attention_mask)
        # tensor = createTensor(local_batch, model)
        if torch.is_tensor(input_ids):
            final_input_ids = torch.cat((input_ids, final_input_ids), 0)
        else:
            print("NOT a tensor")
    print('Create Tensor end time: ' + str(datetime.now() - begin_time_main))
    return final_input_ids

def createTensor(padded, model):
    begin_time_main = datetime.now()
    print("Create Tensor begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    input_ids = torch.tensor(np.array(padded))
    with torch.no_grad():
        last_hidden_states = model(input_ids)
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:, 0, :].numpy()
        print('Create Tensor end time: ' + str(datetime.now() - begin_time_main))
        return features


def padding(tokenized):
    begin_time_main = datetime.now()
    print("Padding begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    print('tokenized length: ' + str(len(tokenized)))
    max_len = 0

    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)


    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])
    # np.array(padded).shape
    print('Padding end time: ' + str(datetime.now() - begin_time_main))
    return padded

def tokenizeText1(df, labels, text_column_name, model_class, tokenizer_class, pretrained_weights):
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
        print('type of padded: ' + str(type(padded)))
        # tensor = batch(padded, labels, model)
        tensor = createTensor(padded, model)
        print('tokenize end time: ', str(datetime.now() - begin_time_main))
        return tensor
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
