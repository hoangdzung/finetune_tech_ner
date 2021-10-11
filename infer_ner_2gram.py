
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import BertForTokenClassification
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

import numpy as np

import re
from pathlib import Path
from tqdm import tqdm 
import argparse
import os  

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
        ngram=1,
        low_level=True
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
        if ngram>1 and low_level:
            sequence_output = (sequence_output+ torch.roll(sequence_output, -1, 1))/2
        logits = self.classifier(sequence_output)
        if ngram>1 and not low_level:
            logits = (logits+ torch.roll(logits, -1, 1))/2
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
parser.add_argument("--pretrained")
parser.add_argument("--ngram", type=int, default=1)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--low_level", action="store_true")
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
model.eval()
while(True):
    sentence = input("Your sentence:")
    tokens = re.findall(r"[\w']+|[.,!?;]", sentence)
    tokens = [token.lower() for token in tokens]

    encodings = tokenizer([tokens], is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
    outputs = model(torch.tensor(encodings['input_ids']).to(device), attention_mask=torch.tensor(encodings['attention_mask']).to(device), ngram=args.ngram, low_level=args.low_level)
    label_indices = torch.argmax(outputs.logits,axis=2)[0][1:-1].cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(encodings.input_ids[0],True)
    print(list(zip(tokens,label_indices)))