import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Read the user reviews from a file
with open('reviews.txt', 'r') as f:
    reviews = f.readlines()

# Pre-process the user reviews
# Convert the reviews to BERT input format
input_ids = tokenizer.batch_encode_plus(
    reviews,
    max_length=128,
    truncation=True,
    padding='max_length',
    return_tensors='pt'
)

# Create a TensorDataset from the input data
data = TensorDataset(**input_ids)

# Create a DataLoader to load the data in batches
loader = DataLoader(data, batch_size=32, shuffle=True)

# Set the model to eval mode
model.eval()

# Initialize an empty list to store the logits
logits_list = []

# Process each batch of data using a for loop
for batch in loader:
    # Compute the logits for the batch
    logits = model(**batch)[0]

    # Append the logits to the logits_list
    logits_list.append(logits)

# Concatenate the logits from each batch
logits = torch.cat(logits_list, dim=0)

# Print the predicted class for each review
preds = logits.argmax(-1)
print(preds)
