
from common import getDataFrame

import numpy as np
import math

import torch
from torch.utils.data import Dataset, DataLoader
import transformers as ppb  # pytorch transformers

from datetime import datetime

class ReviewDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self):
        '''Initialization'''
        # id, text, star rating, label
        # 31679,mine almost burned house,1,1
        df = getDataFrame()
        self.text = df['clean_text']
        self.labels = df['human_tag']
        #self.sub_array = np.array_split(df['clean_text'].to_numpy(), 32) 
        # self.labels = torch.from_numpy(np.array(df['human_tag']))
        self.n_samples = len(df.index)

  def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.text[index], self.labels[index]



def padding(tokenized):
    begin_time_main = datetime.now()
    print("Padding begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    print(type(tokenized))
    print('tokenized length: ' + str(len(tokenized)))
    max_len = 0
    zeros = np.zeros((512,), dtype=int)
    index = 0
    for item in tokenized:
        zeros[index] = item
        index = index + 1
        if index == 511:
            break
    print('Padding end time: ' + str(datetime.now() - begin_time_main))
    return zeros

def createTensor(padded, model):
    begin_time_main = datetime.now()
    print("Create Tensor begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
    # input_ids = torch.from_numpy(padded)
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)
    # attention_mask = torch.from_numpy(attention_mask)

    with torch.no_grad():
        # last_hidden_states = model(input_ids)
        last_hidden_states = model(input_ids, attention_mask=attention_mask)    
        # Slice the output for the first position for all the sequences, take all hidden unit outputs
        features = last_hidden_states[0][:,0,:].numpy()        
        print('Create Tensor end time: ' + str(datetime.now() - begin_time_main))
        return features

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
model = model_class.from_pretrained(pretrained_weights)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)
model.resize_token_embeddings(len(tokenizer))
dataset = ReviewDataset()
first_data = dataset[0]
# features, labels = first_data
# print(features, labels)
dataLoader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False, num_workers=4)

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/1000)

for epoch in range(num_epochs):
    for i, (text, labels) in enumerate(dataLoader):
            # forward backward, update
            
            begin_time_main = datetime.now()
            print("Create Tensor begin time: ", begin_time_main.strftime("%m/%d/%Y, %H:%M:%S"))
            if(i+1) % 2 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}')
            print("0 : ", text[0])
            # tokenized = tokenizer.encode(text, add_special_tokens=True, max_length=511)
            tokenized = text[0].apply((lambda x: tokenizer.encode(x,add_special_tokens=True, max_length=511)))
            padded = padding(tokenized)
            features = createTensor(padded, model)           
            
            print('Duration: ' + str(datetime.now() - begin_time_main))
            print('One iteration COMPLETE')

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(feature, labels)
