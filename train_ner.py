
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

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--early_step", type=int, default=3)
parser.add_argument("--save_dir")
parser.add_argument("--pretrained")
parser.add_argument("--infer_only", action="store_true")
args = parser.parse_args()

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
if args.pretrained is not None:
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained, num_labels=len(unique_tags))
else:
    model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(unique_tags))

model.to(device)

if not args.infer_only:
    model.train()

    texts, tags = read_wnut(args.data)
    print("There are {} sentences in the dataset".format(len(texts)))
    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

    unique_tags = set(tag for doc in tags for tag in doc)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)

    train_labels = encode_tags(train_tags, train_encodings)
    val_labels = encode_tags(val_tags, val_encodings)

    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = WNUTDataset(train_encodings, train_labels)
    val_dataset = WNUTDataset(val_encodings, val_labels)


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
    with open(os.path.join(args.save_dir, 'config.json')) as json_file:
        data = json.load(json_file)
        labelmap = {k:id2tag[v] for k,v in data["label2id"].items()}

else:
    with open(os.path.join(args.save_dir, 'labelmap.json')) as json_file:
        labelmap = json.load(json_file)
        
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

print(labelmap)
with open(os.path.join(args.save_dir, 'labelmap.json'), 'w') as fp:
    json.dump(labelmap, fp)

while(True):
    example = input("Your sentence:")
    if len(example)==0:
        break
    ner_results = nlp(example)
    print([(ner_result['word'], labelmap[ner_result['entity']]) for ner_result in ner_results])