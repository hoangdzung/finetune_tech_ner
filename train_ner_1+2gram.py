
import torch
from torch import nn
from torch.utils.data import DataLoader 
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import BertForTokenClassification
from transformers import pipeline
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
import numpy as np

import re
from pathlib import Path
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
import argparse
import json
import pandas as pd
import os 
from sklearn.metrics import f1_score,accuracy_score, recall_score, precision_score
from data_utils import encode_tags_masks
 
def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag1_docs = []
    tag2_docs = []
    for doc in raw_docs:
        tokens = []
        tags1 = []
        tags2 = []
        for line in doc.split('\n'):
            token, tag2, tag1 = line.split('\t')
            tokens.append(token.lower())
            tags1.append(tag1)
            tags2.append(tag2)
        token_docs.append(tokens)
        tag1_docs.append(tags1)
        tag2_docs.append(tags2)

    return token_docs, tag1_docs, tag2_docs

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels1, labels2):
        self.encodings = encodings
        self.labels1 = labels1
        self.labels2 = labels2

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels1'] = torch.tensor(self.labels1[idx])
        item['labels2'] = torch.tensor(self.labels2[idx])
        return item

    def __len__(self):
        return len(self.labels1)

class TwoHeadTokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None
    logits1: torch.FloatTensor = None
    logits2: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class NGramBertForTokenClassification(BertForTokenClassification):

    def __init__(self, config):
        super().__init__(config)
        #self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)
        #self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels1=None,
        labels2=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None        
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
        sequence_output1 = self.dropout(sequence_output)
        sequence_output2 = self.dropout((sequence_output+ torch.roll(sequence_output, -1, 1))/2)

        logits1 = self.classifier(sequence_output1)
        logits2 = self.classifier(sequence_output2)

        loss1, loss2 = None, None
        if labels1 is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits1 = logits1.view(-1, self.num_labels)
                active_labels1 = torch.where(
                    active_loss, labels1.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels1)
                )
                loss1 = loss_fct(active_logits1, active_labels1)
            else:
                loss1 = loss_fct(logits1.view(-1, self.num_labels), labels1.view(-1))
        if labels2 is not None:
            loss_fct2 = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits2 = logits2.view(-1, self.num_labels)
                active_labels2 = torch.where(
                    active_loss, labels2.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels2)
                )
                loss2 = loss_fct2(active_logits2, active_labels2)
            else:
                loss2 = loss_fct2(logits2.view(-1, self.num_labels), labels2.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TwoHeadTokenClassifierOutput(
            loss1=loss1,
            loss2=loss2,
            logits1=logits1,
            logits2=logits2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def evaluate(model, dataloader):
    model.eval()
    true_labels1, pred_labels1 = [], []
    true_labels2, pred_labels2 = [], []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels1 = batch['labels1'].to(device)
        labels2 = batch['labels2'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)
        
        label_indices1 = torch.argmax(outputs.logits1,axis=2)
        true_labels_1 = labels1[labels1!=-100].cpu().numpy().tolist()
        pred_labels_1 = label_indices1[labels1!=-100].detach().cpu().numpy().tolist()

        label_indices2 = torch.argmax(outputs.logits2,axis=2)
        true_labels_2 = labels2[labels2!=-100].cpu().numpy().tolist()
        pred_labels_2 = label_indices2[labels2!=-100].detach().cpu().numpy().tolist()

        true_labels1 += true_labels_1
        pred_labels1 += pred_labels_1 
        
        true_labels2 += true_labels_2
        pred_labels2 += pred_labels_2 
    return (f1_score(true_labels1, pred_labels1,pos_label=0),\
        recall_score(true_labels1, pred_labels1,pos_label=0),\
        precision_score(true_labels1, pred_labels1,pos_label=0)),\
            (f1_score(true_labels2, pred_labels2,pos_label=0),\
        recall_score(true_labels2, pred_labels2,pos_label=0),\
        precision_score(true_labels2, pred_labels2,pos_label=0)) 

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
parser.add_argument("--weight1", type=float, default=1.0)
parser.add_argument("--weight2", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--mturk", action="store_true")
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
    texts, tags1, tags2 = pickle.load(open(args.data,'rb'))
else:
    texts, tags1, tags2 = read_wnut(args.data)
print("There are {} sentences in the dataset".format(len(texts)))

if args.test:
    if args.test.endswith('pkl'):
        import pickle
        test_texts, test_tags1, test_tags2 = pickle.load(open(args.test,'rb'))
    else:
        test_texts, test_tags1, test_tags2 = read_wnut(args.test)
    train_texts, val_texts, train_tags1, val_tags1, train_tags2, val_tags2= train_test_split(texts, tags1, tags2, test_size=args.val_frac,random_state=args.seed)
 
else:
    train_texts, test_texts, train_tags1, test_tags1, train_tags2, test_tags2 = train_test_split(texts, tags1, tags2, test_size=args.test_frac,random_state=args.seed)
    train_texts, val_texts, train_tags1, val_tags1, train_tags2, val_tags2 = train_test_split(train_texts, train_tags1, train_tags2, test_size=args.val_frac/(1-args.test_frac),random_state=args.seed)

unique_tags = sorted(list(set(tag for doc in tags1+tags2 for tag in doc)))
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)

train_labels1 = encode_tags_masks(train_tags1, train_encodings,tag2id,masks=False)
val_labels1 = encode_tags_masks(val_tags1, val_encodings,tag2id,masks=False)
test_labels1 = encode_tags_masks(test_tags1, test_encodings,tag2id,masks=False)

train_labels2 = encode_tags_masks(train_tags2, train_encodings,tag2id,masks=False)
val_labels2 = encode_tags_masks(val_tags2, val_encodings,tag2id,masks=False)
test_labels2 = encode_tags_masks(test_tags2, test_encodings,tag2id,masks=False)

train_encodings.pop("offset_mapping") 
val_encodings.pop("offset_mapping") 
test_encodings.pop("offset_mapping") 
train_dataset = WNUTDataset(train_encodings, train_labels1, train_labels2)
val_dataset = WNUTDataset(val_encodings, val_labels1, val_labels2)
test_dataset = WNUTDataset(test_encodings, test_labels1, test_labels2)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

#optim = AdamW(list(model.classifier.parameters())+list(model.classifier2.parameters()), lr=args.lr)
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
        labels1 = batch['labels1'].to(device)
        labels2 = batch['labels2'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels1=labels1, labels2=labels2)
        loss1 = outputs[0]
        loss2 = outputs[1]
        loss = args.weight1*loss1 +args.weight2*loss2
        loss.backward()
        total_loss += loss.item()
        optim.step()
    
    (val_f11, val_rc1, val_pr1),(val_f12, val_rc2, val_pr2) = evaluate(model, val_loader)
    val_f1 = (args.weight1*val_f11+args.weight2*val_f12)/(args.weight1+args.weight2)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        n_wo_progress = 0
        model.save_pretrained(args.save_dir)
    else:
        n_wo_progress+=1
    print("Epoch {}, train loss {}, val_f11 {}, val_f12 {}, val_f1 {} , best val_f1 {}, #step without progress {}"\
    .format(epoch, total_loss, val_f11, val_f12, val_f1, best_val_f1, n_wo_progress))
    print(val_rc1, val_pr1)
    print(val_rc2, val_pr2)
    if n_wo_progress > args.early_step:
        print("Early stop")
        break


model = NGramBertForTokenClassification.from_pretrained(args.save_dir).to(device)

(test_f11, test_rc1, test_pr1),(test_f12, test_rc2, test_pr2) = evaluate(model, test_loader)
print(test_f11, test_rc1, test_pr1)
print(test_f12, test_rc2, test_pr2)
    
