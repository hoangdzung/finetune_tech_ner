
import torch
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import pipeline
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

import numpy as np

import re
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import argparse
import json 
import os 

from data_utils import read_wnut, encode_tags_masks, WNUTDataset, offset_to_biluo, offset_to_bio
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--train_data")
parser.add_argument("--test_data")
parser.add_argument("--encode_format")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--early_step", type=int, default=3)
parser.add_argument("--save_dir")
parser.add_argument("--pretrained")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
if args.pretrained is not None:
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained, num_labels=len(args.encode_format))
else:
    model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(args.encode_format))

model.to(device)

model.train()

train_texts, train_tags = offset_to_biluo(args.train_data)
print("There are {} sentences in the train dataset".format(len(train_tags)))
test_texts, test_tags = offset_to_biluo(args.test_data)
print("There are {} sentences in the test dataset".format(len(test_tags)))
# train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=.2)


unique_tags = args.encode_format
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
train_labels = encode_tags_masks(train_tags, train_encodings,tag2id,masks=False)
train_encodings.pop("offset_mapping") # we don't want to pass this to the model
train_dataset = WNUTDataset(train_encodings, train_labels)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
test_labels = encode_tags_masks(test_tags, test_encodings,tag2id,masks=False)
test_encodings.pop("offset_mapping") # we don't want to pass this to the model
test_dataset = WNUTDataset(test_encodings, test_labels)
# test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

training_args = TrainingArguments(
    output_dir=args.save_dir,          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

trainer.train()

    # optim = AdamW(model.parameters(), lr=args.lr)

#     best_loss = 10e10
#     for epoch in range(args.epochs):
#         total_loss = 0
#         for batch in tqdm(train_loader):
#             optim.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs[0]
#             loss.backward()
#             total_loss += loss.item()
#             optim.step()
#         if total_loss < best_loss:
#             best_loss = total_loss
#             n_wo_progress = 0
#             model.save_pretrained(args.save_dir)
#         else:
#             n_wo_progress+=1
#         print("Epoch {}, current loss {}, best loss {}, #step without progress {}".format(epoch, total_loss, best_loss, n_wo_progress))

#         if n_wo_progress > args.early_step:
#             print("Early stop")
#             break

#     model = AutoModelForTokenClassification.from_pretrained(args.save_dir)
#     model.eval()
#     if not args.binary_label:
#         with open(os.path.join(args.save_dir, 'config.json')) as json_file:
#             data = json.load(json_file)
#             labelmap = {k:id2tag[v] for k,v in data["label2id"].items()}
#         print(labelmap)
#         with open(os.path.join(args.save_dir, 'labelmap.json'), 'w') as fp:
#             json.dump(labelmap, fp)

# elif not args.binary_label:
#     with open(os.path.join(args.save_dir, 'labelmap.json')) as json_file:
#         labelmap = json.load(json_file)
#     print(labelmap)

# nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# while(True):
#     example = input("Your sentence:")
#     if len(example)==0:
#         break
#     ner_results = nlp(example)
#     if args.binary_label:
#         print([(ner_result['word'], ner_result['entity'], ner_result['score']) for ner_result in ner_results])
#     else:
#         print([(ner_result['word'], labelmap[ner_result['entity']], ner_result['score']) for ner_result in ner_results])