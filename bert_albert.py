import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AlbertModel

# load the pre-trained BERT model and the tokenizer
bert_model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# load the pre-trained ALBERT model
albert_model = AlbertModel.from_pretrained("albert-base-v1")

# define the review input file
review_file = "reviews.txt"

# read the reviews from the file
with open(review_file, "r") as f:
    reviews = f.readlines()

# pre-process the reviews to prepare them for input to the model
# convert to lowercase
reviews = [review.lower() for review in reviews]
# strip leading and trailing whitespace
reviews = [review.strip() for review in reviews]

# tokenize the reviews
tokenized_reviews = [tokenizer.tokenize(review) for review in reviews]
# convert the tokenized reviews to BERT inputs
bert_inputs = [tokenizer.convert_tokens_to_ids(review) for review in tokenized_reviews]

# create masks to indicate which tokens are padding and which are not
padding_mask = [torch.tensor(review, dtype=torch.long) for review in bert_inputs]

# use the fine-tuned BERT model, the fine-tuned ALBERT model, and the CNN model to classify the reviews as containing safety concerns or not
results = []
for review, padding_mask in zip(bert_inputs, padding_mask):
    bert_inputs = torch.tensor([review])
    bert_output = bert_model(bert_inputs, attention_mask=padding_mask)
    bert_logits = bert_output[0]
    bert_prediction = torch.sigmoid(bert_logits)

    albert_inputs = torch.tensor([review])
    albert_output = albert_model(albert_inputs, attention_mask=padding_mask)
    albert_logits = albert_output[0]
    albert_prediction = torch.sigmoid(albert_logits)
    
    # combine the predictions from the BERT model, the ALBERT model, and the CNN model using a weighted average
    bert_weight = 0.5
    albert_weight = 0.3
    prediction = bert_weight * bert_prediction + albert_weight * albert_prediction
    if prediction[0][0] > 0.5:
        results.append("Safety Concern")
    else:
        results.append("No Safety Concern")
    
    

# print the results
for review, result in zip(reviews, results):
    print("Review:", review)
    print("Result:", result)
    print()
