
import torch
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import BertForTokenClassification
from transformers import pipeline
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput

import numpy as np

import re
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import argparse
import json
import pandas as pd
import os 

from data_utils import read_mturk, encode_tags_masks, encode_mask, MturkDataset, chunk_based_tokenize
 

class PhraseBertForTokenClassification(BertForTokenClassification):

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        phrase_mask=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if phrase_mask is not None:
            sequence_output = torch.swapaxes(
                                torch.matmul(
                                torch.swapaxes(sequence_output, 1,2), phrase_mask.float()
                                ), 2,1)
            sequence_output=sequence_output/torch.unsqueeze(phrase_mask.sum(2),2)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

if args.pretrained is not None:
    model = PhraseBertForTokenClassification.from_pretrained(args.pretrained)
else:
    model = PhraseBertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased')

model.to(device)

if not args.infer_only:
    model.train()
    texts, tags = read_mturk(args.data)
    print("There are {} sentences in the dataset".format(len(texts)))
    
    unique_tags = sorted(list(set(tag for doc in tags for tag in doc)))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    train_encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    train_labels, train_masks = encode_tags_masks(tags, train_encodings,tag2id,masks=True)
    
    train_encodings.pop("offset_mapping") # we don't want to pass this to the model
    train_dataset = MturkDataset(train_encodings, train_labels, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.lr)
    
    best_loss = 10e10
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            phrase_mask = batch['phrase_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, phrase_mask=phrase_mask)
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
    model = PhraseBertForTokenClassification.from_pretrained(args.save_dir).to(device)
    model.eval()
    
while(True):
    example = input("Your sentence:")
    if len(example)==0:
        break
    tokens = chunk_based_tokenize(example)
    encodings = tokenizer([tokens], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    phrase_masks = [encode_mask(encodings.offset_mapping[0])]
    
    encodings.pop("offset_mapping") # we don't want to pass this to the model
    outputs = model(torch.tensor(encodings['input_ids']).to(device), 
                attention_mask=torch.tensor(encodings['attention_mask']).to(device),
                phrase_mask=torch.tensor(phrase_masks).to(device))
    label_indices = np.argmax(outputs.logits.detach().to('cpu').numpy(),axis=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])
    phrase_mask = np.array(phrase_masks[0]).sum(0)
    squeeze_label_indices = []
    start_id=0
    while(start_id<len(label_indices)):
        squeeze_label_indices.append(label_indices[start_id])
        start_id += phrase_mask[start_id]
    print(list(zip(tokens, label_indices))[1:-1])