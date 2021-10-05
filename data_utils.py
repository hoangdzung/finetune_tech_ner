from pathlib import Path
import pandas as pd
import re
import numpy as np
import torch
import spacy 
import json

from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
try:
    nlp = spacy.load("en_core_web_sm")
    print("en_core_web_sm loaded")
except:
    nlp = spacy.load("en_core_web_trf")
    print("en_core_web_trf loaded")

def read_wnut(file_path, binary_label=True):
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
            if binary_label and tag=='I':
                tag ='B'
            tokens.append(token.lower())
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def read_wnut_phrase(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        sub_tokens = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            if tag == 'O':
                if len(sub_tokens) > 0:
                    tokens.append(" ".join(sub_tokens))
                    tags.append("B")
                tokens.append(token.lower())
                tags.append(tag)
            else:
                sub_tokens.append(token.lower())
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def read_mturk(file_path):
    df = pd.read_csv(file_path)
    df = df[['Input.sentence','Answer.entity']]
    token_docs = []
    tag_docs = []
    for _, row in tqdm(df.iterrows(), desc="Read raw data..."):
        data = json.loads(row['Input.sentence'])
        token_id = list(map(int, row['Answer.entity'].split(",")))
        for json_sen in data:
            tokens = []
            tags = []
            for json_token in json_sen:

                token = " ".join(json_token['tokens'])
                if token=='\xa0':
                    continue
                tag = 'B' if len(set(json_token['indexes']).intersection(token_id)) >0 else 'O'
                tokens.append(token.lower())
                tags.append(tag)
            token_docs.append(tokens)
            tag_docs.append(tags)
    return token_docs, tag_docs

def encode_mask(offset_mappings):
    pair = []
    mask=np.zeros((len(offset_mappings),len(offset_mappings)))
    for idx,i in enumerate(offset_mappings):
        if i[0]==0:
            if idx!=0:
                mask[pair[0]:pair[1]+1,pair[0]:pair[1]+1] =1
            pair = [idx,idx]
        else:
            pair[-1]=idx
    mask[pair[0]:pair[1]+1,pair[0]:pair[1]+1] =1
    return mask.tolist()

def encode_tags_masks(tags, encodings, tag2id, masks=False):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    encoded_masks = []
    for idx, (doc_labels, doc_offset) in tqdm(enumerate(zip(labels, encodings.offset_mapping)),desc="Encode masks and labels"):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        try:
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        except:
            print(idx) 
        else:  
            encoded_labels.append(doc_enc_labels.tolist())
            if masks:
                encoded_masks.append(encode_mask(doc_offset))
    if masks:
        return encoded_labels,encoded_masks 
    else:
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


class MturkDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, masks):
        self.encodings = encodings
        self.labels = labels
        self.masks = masks

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['phrase_mask'] = torch.tensor(self.masks[idx])
        return item

    def __len__(self):
        return len(self.labels)

def chunk_based_tokenize(sen):
    doc=nlp(sen)

    id_map = {} # x_1:[x_1,x_2,...,x_n], x_2 :False 
    for chunk in doc.noun_chunks:
        prev_accepted = False
        for token in chunk:
            if not token.is_stop and not prev_accepted:
                start = token.i
                id_map[start]=[token.text.lower()]
                prev_accepted=True
            elif not token.is_stop and prev_accepted:
                id_map[token.i] = False
                id_map[start].append(token.text.lower())
                prev_accepted=True
            elif token.is_stop and prev_accepted:
                prev_accepted=False

    sen_data = []
    for token in doc:
        if token.i not in id_map:
            sen_data.append(token.text.lower())
        elif id_map[token.i]:
            sen_data.append(' '.join(id_map[token.i]))
    return sen_data