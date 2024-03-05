from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoModelForSequenceClassification, BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from torch.optim import AdamW
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    #First element of model_output contains all token embeddings, same thing as last hidden state
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# Sentences we want sentence embeddings for

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model
# Tokenize sentences
sentences = ['THIS is an EXAMPLE sentence', 
            'This is another sentence that we intentionally made longer with a misssspelling',
            'Sometimes material should be about unrelated subjects like hiking']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


encoded_input
tokenizer.vocab_size
for k, v, in tokenizer.vocab.items():
    if v in [101, 102, 2023, 15734, 30521, 11880, 2989, 4757, 3335]:
        print(k, v)
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)


model_output.last_hidden_state.size(), model_output.pooler_output.size()
model_output[0], "last hidden state: ", model_output.last_hidden_state
# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)


sentence_embeddings, "pooled output ", model_output.pooler_output

encoded_input['attention_mask'].size(),encoded_input['attention_mask']
encoded_input['attention_mask'].unsqueeze(-1).size(), encoded_input['attention_mask'].unsqueeze(-1)
input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(model_output[0].size()).float()
input_mask_expanded.size(), input_mask_expanded
model_output[0] * input_mask_expanded
summed = torch.sum(model_output[0] * input_mask_expanded, 1)
summed.size(), summed
torch.clamp(input_mask_expanded.sum(1), min=1e-9)
torch.sum(model_output[0] * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
query = ["I need an example", 
         "I need a sample",
         "Did you purposefully make an grammar error"]

encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    query_output = model(**encoded_query)
from torch.nn import CosineSimilarity
cos = CosineSimilarity(dim=1, eps=1e-6)
output_0 = cos(query_output.pooler_output[0], model_output.pooler_output)
output_1 = cos(query_output.pooler_output[1], model_output.pooler_output)
output_2 = cos(query_output.pooler_output[2], model_output.pooler_output)
output_0, output_1, output_2
with open('time_policy_sentences.txt') as t:
    time_policy_text = t.readlines()
t.close()

with open('travel_policy_sentences.txt') as t:
    travel_policy_text = t.readlines()
t.close()
time_policy_labels = [0]*len(time_policy_train_text)
travel_policy_labels = [1]*len(travel_policy_train_text)
list(zip(time_policy_text, time_policy_labels))
list(zip(travel_policy_text, travel_policy_labels))
all_text = travel_policy_text + time_policy_text
all_labels = travel_policy_labels + time_policy_labels
list(zip(all_text,all_labels))
train_encodings = tokenizer(all_text, truncation=True, padding=True, return_tensors='pt')
train_labels = all_labels
class policy_dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = policy_dataset(train_encodings, train_labels)
train_batchsize = 2 # so there will be 28,800/64 = 450 minibatches
#val_batchsize = 32 # so there will be 3500/32 = 100 minibatches

train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=val_batchsize, shuffle=True)
# 2 labels for time and travel
# eventually this will be 45 or so
model_class = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', num_labels=2)
model_class 
torch.cuda.is_available()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_class.to(device)
print('training on {}'.format(model_class.device))

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):
    for i, batch in enumerate(train_loader):
        model_class.train()
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        print('labels ' , labels)
        outputs = model_class(input_ids, attention_mask=attention_mask, labels=labels)
        print(outputs)
        loss = outputs[0]
        _, predicted = torch.max(outputs.logits, dim=1)
        accuracy = sum(predicted == labels)/train_batchsize
        if i % 10 == 0:
            print('epoch {}, minibatch {}: training loss = {} accuracy = {}'.format(epoch, i, loss, accuracy))
        loss.backward()
        optim.step()
        
    # check performance on validation dataset
    for k, val_batch in enumerate(val_loader):
        model_class.eval()
        val_input_ids = val_batch['input_ids'].to(device)
        val_attention_mask = val_batch['attention_mask'].to(device)
        val_labels = val_batch['labels'].to(device)
        #print('labels ' , val_labels)
        val_outputs = model_class(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
        #print(outputs)
        val_loss = val_outputs[0]
        _, val_predicted = torch.max(val_outputs.logits, dim=1)
        val_accuracy = sum(val_predicted == val_labels)/val_batchsize
        if k % 50 == 0:
            print('epoch {}, minibatch {}: validation loss = {} accuracy = {}'.format(epoch, k, val_loss, val_accuracy))
    
    print('epoch {} validation loss = {} accuracy = {}'.format(epoch, val_loss))
        
        
    model_class.save('miniLM_fresh_class_head_epoch{}'.format(epoch))
model_class.eval()

total_correct = 0
total = 0
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
for test_batch in test_loader:
    test_input_ids = test_batch['input_ids'].to(device)
    test_attention_mask = test_batch['attention_mask'].to(device)
    test_labels = test_batch['labels'].to(device)
    test_outputs = model_class(test_input_ids, attention_mask=test_attention_mask, labels=test_labels)
    _, test_predicted = torch.max(test_outputs.logits, dim=1)
    total_correct += sum(test_predicted == test_labels)
    total += len(test_labels)
total_accuracy = total_correct/total
model_finetuned = AutoModel('miniLM_fresh_class_head_epoch3')
model_finetuned
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

encoded_input
# Compute token embeddings
with torch.no_grad():
    output = model(**encoded_input)
    #finetuned_output = model_finetunes(**enconded_input)

# this should get us the hidden state of the word theater
output.last_hidden_state[1][3], finetuned_output.last_hidden_state[1][3]
