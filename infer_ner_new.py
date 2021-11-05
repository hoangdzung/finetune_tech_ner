
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

from data_utils import read_wnut, encode_tags_masks, WNUTDataset, offset_to_biluo
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--test_data")
parser.add_argument("--out")
parser.add_argument("--pretrained")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModelForTokenClassification.from_pretrained(args.pretrained, num_labels=5)

model.to(device)

model.eval()

test_texts, test_tags = offset_to_biluo(args.test_data)
print("There are {} sentences in the test dataset".format(len(texts)))
# train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=.2)

predictions = []
for text in tqdm(test_texts):
    train_encodings = tokenizer.encode(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(train_encodings[0])
    preds = torch.argmax(model(train_encodings).logits,-1)[0].detach().cpu().numpy().tolist()
    tech_terms = ""
    for token, pred in zip(tokens, preds):
        if pred != 4:
            tech_terms+=" "+token 
        elif tech_terms[-1]!='#':
            tech_terms+="#"
    tech_terms = tech_terms.strip("#").split("#")
    predictions.append([text, tech_terms])

with open(args.out,'w') as f:
    json.dump(predictions, f)