
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
from sklearn.metrics import f1_score,accuracy_score
from data_utils import read_wnut, WNUTDataset,encode_tags_masks
 

class NGramBertForTokenClassification(BertForTokenClassification):

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
        ngram=1
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
        sequence_output = self.dropout(sequence_output)
        if ngram>1:
            sequence_output = (sequence_output+ torch.roll(sequence_output, -1, 1))/2
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

def evaluate(model, dataloader, ngram=1, eval_norm=True):
    model.eval()
    true_labels, pred_labels = [], []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, ngram=ngram)
        label_indices = torch.argmax(outputs.logits,axis=2)
        true_labels_ = labels[labels!=-100].cpu().numpy().tolist()
        pred_labels_ = label_indices[labels!=-100].detach().cpu().numpy().tolist()
        if not eval_norm and ngram==1:
            for i in range(len(true_labels_)-1):
                if true_labels_[i]==0 and true_labels_[i+1]==0:
                    true_labels.append(0)
                else:
                    true_labels.append(0)
                if pred_labels_[i] == 0 and pred_labels_[i+1] == 0:
                    pred_labels.append(0)
                else:
                    pred_labels.append(1)
        else:
            true_labels += true_labels_ 
            pred_labels += pred_labels_ 
    return f1_score(true_labels, pred_labels,pos_label=0), accuracy_score(true_labels, pred_labels)

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--test")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--early_step", type=int, default=3)
parser.add_argument("--save_dir")
parser.add_argument("--pretrained")
parser.add_argument("--test_frac",type=float, default=0.2)
parser.add_argument("--val_frac",type=float, default=0.2)
parser.add_argument("--ngram", type=int, default=1)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--mturk", action="store_true")
parser.add_argument("--norm_eval", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

if args.pretrained is not None:
    model = NGramBertForTokenClassification.from_pretrained(args.pretrained)
else:
    model = NGramBertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased')

model.to(device)

if args.data.endswith('pkl'):
    import pickle
    texts, tags = pickle.load(open(args.data,'rb'))
else:
    texts, tags = read_wnut(args.data)
print("There are {} sentences in the dataset".format(len(texts)))

if args.test:
    if args.data.endswith('pkl'):
        import pickle
        test_texts, test_tags = pickle.load(open(args.data,'rb'))
    else:
        test_texts, test_tags = read_wnut(args.data)
    train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=args.val_frac,random_state=args.seed)
 
else:
    train_texts, test_texts, train_tags, test_tags = train_test_split(texts, tags, test_size=args.test_frac,random_state=args.seed)
    train_texts, val_texts, train_tags, val_tags = train_test_split(train_texts, train_tags, test_size=args.val_frac/(1-args.test_frac),random_state=args.seed)

unique_tags = sorted(list(set(tag for doc in tags for tag in doc)))
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
train_labels = encode_tags_masks(train_tags, train_encodings,tag2id,masks=False)
val_labels = encode_tags_masks(val_tags, val_encodings,tag2id,masks=False)
test_labels = encode_tags_masks(test_tags, test_encodings,tag2id,masks=False)

train_encodings.pop("offset_mapping") 
val_encodings.pop("offset_mapping") 
test_encodings.pop("offset_mapping") 
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)
test_dataset = WNUTDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

optim = AdamW(model.parameters(), lr=args.lr)

best_val_f1 = 0
n_wo_progress = 0
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, ngram=args.ngram)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        optim.step()
    
    val_f1, val_acc = evaluate(model, val_loader, args.ngram)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        n_wo_progress = 0
        model.save_pretrained(args.save_dir)
    else:
        n_wo_progress+=1
    print("Epoch {}, train loss {}, val_f1 {} , best val_f1 {}, #step without progress {}".format(epoch, total_loss, val_f1, best_val_f1, n_wo_progress))
    if n_wo_progress > args.early_step:
        print("Early stop")
        break


model = NGramBertForTokenClassification.from_pretrained(args.save_dir).to(device)

test_f1, test_acc = evaluate(model, test_loader, args.ngram)
print(test_f1, test_acc)
    