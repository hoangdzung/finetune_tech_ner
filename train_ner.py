
import torch
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import pipeline

import numpy as np

import re
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import argparse
import json 
import os 

from data_utils import read_wnut, encode_tags_masks, WNUTDataset

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--early_step", type=int, default=3)
parser.add_argument("--save_dir")
parser.add_argument("--pretrained")
parser.add_argument("--binary_label", action="store_true")
parser.add_argument("--infer_only", action="store_true")
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
if args.binary_label:
    NUM_LABELS=2
else:
    NUM_LABELS=3
if args.pretrained is not None:
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained, num_labels=NUM_LABELS)
else:
    model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=NUM_LABELS)

model.to(device)

if not args.infer_only:
    model.train()

    texts, tags = read_wnut(args.data, args.binary_label)
    print("There are {} sentences in the dataset".format(len(texts)))

    unique_tags = sorted(list(set(tag for doc in tags for tag in doc)))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    train_encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    train_labels = encode_tags_masks(tags, train_encodings,tag2id,masks=False)
    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    train_dataset = WNUTDataset(train_encodings, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optim = AdamW(model.parameters(), lr=args.lr)

    best_loss = 10e10
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            optim.step()
        if total_loss < best_loss:
            best_loss = total_loss
            n_wo_progress = 0
            model.save_pretrained(args.save_dir)
        else:
            n_wo_progress+=1
        print("Epoch {}, current loss {}, best loss {}, #step without progress {}".format(epoch, total_loss, best_loss, n_wo_progress))

        if n_wo_progress > args.early_step:
            print("Early stop")
            break

    model = AutoModelForTokenClassification.from_pretrained(args.save_dir)
    model.eval()
    if not args.binary_label:
        with open(os.path.join(args.save_dir, 'config.json')) as json_file:
            data = json.load(json_file)
            labelmap = {k:id2tag[v] for k,v in data["label2id"].items()}
        print(labelmap)
        with open(os.path.join(args.save_dir, 'labelmap.json'), 'w') as fp:
            json.dump(labelmap, fp)

elif not args.binary_label:
    with open(os.path.join(args.save_dir, 'labelmap.json')) as json_file:
        labelmap = json.load(json_file)
    print(labelmap)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

while(True):
    example = input("Your sentence:")
    if len(example)==0:
        break
    ner_results = nlp(example)
    if args.binary_label:
        print([(ner_result['word'], ner_result['entity'], ner_result['score']) for ner_result in ner_results])
    else:
        print([(ner_result['word'], labelmap[ner_result['entity']], ner_result['score']) for ner_result in ner_results])