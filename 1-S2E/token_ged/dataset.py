import random
import math
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer
import json

class Dataset():
    def __init__(
        self,
        tokens: List[List[str]],
        labels,
        tokenizer: PreTrainedTokenizer,
        max_len: int
    ) -> None:
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx: int) -> dict:
        tokens = self.tokens[idx]
        encode = self.tokenizer.batch_encode_plus(
            [tokens],
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return_dict = {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
        }
        label = self.labels[idx]
        return_dict['labels'] = torch.tensor(label)
        return return_dict

    def __len__(self):
        return len(self.tokens)
    
def convert_subword_labels(tokens, labels, tokenizer, max_len):
    subword_labels = []
    for token, label in zip(tokens, labels):
        encode = tokenizer(
            token,
            is_split_into_words=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        prev_id = None
        sub_label = []
        for wid in encode.word_ids(0):
            if wid is None:
                sub_label.append(-100)
            elif prev_id != wid:
                sub_label.append(label[wid])
            else:
                sub_label.append(-100)
            prev_id = wid
        subword_labels.append(sub_label)
    return subword_labels

def generate_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int
) -> Dataset:
    '''
    This function recieves input file path(s) and returns a Dataset instance.
    '''
    srcs = []
    labels = []
    data = json.load(open(input_file))
    for d in data['dataset'].values():
        srcs.append(d['src'])
        labels.append(d['label'])
    tokens = [s.split(' ') for s in srcs]
    subword_labels = convert_subword_labels(
        tokens, labels, tokenizer, max_len
    )

    return Dataset(
        tokens=tokens,
        labels=subword_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

def generate_dataset_for_inference(
    srcs,
    tokenizer,
    max_len
):
    tokens = [s.split(' ') for s in srcs]
    labels = [len(t) * [0] for t in tokens]
    subword_labels = convert_subword_labels(
        tokens, labels, tokenizer, max_len
    )
    return Dataset(
        tokens=tokens,
        labels=subword_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )