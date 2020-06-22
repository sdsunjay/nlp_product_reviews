
from common import getDataFrame

import numpy as np
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datetime import datetime
import tensorflow as tf
import collections
import json
import os
import pandas as pd
import csv
from transformers import DistilBertTokenizer

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# We set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 512
DATA_COLUMN = 'clean_text'
LABEL_COLUMN = 'human_tag'
LABEL_VALUES = [0, 1]

label_map = {}
for (i, label) in enumerate(LABEL_VALUES):
    label_map[label] = i


class InputFeatures(object):
  """BERT feature vectors."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id

class Input(object):
    """A single training/test input for sequence classification."""

    def __init__(self, text, label=None):
        self.text = text
        self.label = label


def convert_input(text_input):
    # First, we need to preprocess our data so that it matches the data BERT was trained on:
    # 1. Lowercase our text (if we're using a BERT lowercase model)
    # 2. Tokenize it (i.e. "sally says hi" -> ["sally", "says", "hi"])
    # 3. Break words into WordPieces (i.e. "calling" -> ["call", "##ing"])
    #
    # Fortunately, the Transformers tokenizer does this for us!

    tokens = tokenizer.tokenize(text_input.text)
    # print('**tokens**\n{}\n'.format(tokens))

    encode_plus_tokens = tokenizer.encode_plus(text_input.text,
                                               pad_to_max_length=True,
                                               max_length=MAX_SEQ_LENGTH)

    input_ids = encode_plus_tokens['input_ids']
    input_mask = encode_plus_tokens['attention_mask']
    segment_ids = [0] * MAX_SEQ_LENGTH

    label_id = label_map[text_input.label]

    features = InputFeatures(input_ids=input_ids, input_mask=input_mask,
            segment_ids=segment_ids, label_id=label_id)

    # print('**input_ids**\n{}\n'.format(features.input_ids))
    # print('**input_mask**\n{}\n'.format(features.input_mask))
    # print('**segment_ids**\n{}\n'.format(features.segment_ids))
    # print('**label_id**\n{}\n'.format(features.label_id))

    return features


# We'll need to transform our data into a format that BERT understands.
# - `text` is the text we want to classify, which in this case, is the `Request` field in our Dataframe.
# - `label` is the star_rating label (1, 2, 3, 4, 5) for our training input data
def transform_inputs_to_tfrecord(inputs):
    tf_records = []
    for (input_idx, text_input) in enumerate(inputs):
      if input_idx % 10000 == 0:
          print('Writing input {} of {}\n'.format(input_idx, len(inputs)))

      features = convert_input(text_input)

      all_features = collections.OrderedDict()
      all_features['input_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_ids))
      all_features['input_mask'] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.input_mask))
      all_features['segment_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=features.segment_ids))
      all_features['label_ids'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features.label_id]))

      tf_record = tf.train.Example(features=tf.train.Features(feature=all_features))
      tf_records.append(tf_record.SerializeToString())

    return tf_records

def main():
    df = getDataFrame()

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    inputs = df.apply(lambda x: Input(text = x[DATA_COLUMN], label = x[LABEL_COLUMN]), axis = 1)

    tf_records = transform_inputs_to_tfrecord(inputs)

if __name__ == "__main__":
    main()
