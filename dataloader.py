#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from utils import list2tuple, tuple2list, flatten
from transformers import BertTokenizer

# Initialize tokenizer globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(id2rel, id2ent_title):
    id2rel_lst = [id2rel[i] for i in id2rel]
    id2ent_title_lst = [id2ent_title[i] for i in id2ent_title]
    id2rel = tokenizer(id2rel_lst,
                       return_tensors='pt',
                       padding=True, 
                       truncation=True,
                       max_length=512)
    id2ent_title = tokenizer(id2ent_title_lst,
                       return_tensors='pt',
                       padding=True, 
                       truncation=True,
                       max_length=512)
    return id2rel, id2ent_title

class TrainDataset(Dataset):
    def __init__(self, queries, nprod, nentity, nrelation, negative_sample_size, answer, dataset, id2text_queries, text_symbolic_queries2id):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nprod = nprod
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer
        self.dataset = dataset
        self.id2text_queries = id2text_queries
        self.text_symbolic_queries2id = text_symbolic_queries2id

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[(query[0],self.id2text_queries[query[1]])]))
        subsampling_weight = self.count[query] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        negative_sample_list = []
        negative_sample_size = 0
        while negative_sample_size < self.negative_sample_size:
            if self.dataset == 'prime':
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            elif self.dataset == 'amazon':
                negative_sample = np.random.randint(self.nprod, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.answer[query], 
                assume_unique=True, 
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.tensor(negative_sample.tolist())
        positive_sample = torch.LongTensor([tail])

        return positive_sample, negative_sample, subsampling_weight, self.text_symbolic_queries2id[query[0]], flatten(query[0]), query[1], query_structure
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        text_symbolic_query_idx = [_[3] for _ in data]
        symbolic_query = [_[4] for _ in data]
        textual_query = [_[5] for _ in data] #tokenizer([_[4] for _ in data], return_tensors='pt', padding=True, truncation=True, max_length=512)
        query_structure = [_[6] for _ in data]
        return positive_sample, negative_sample, subsample_weight, symbolic_query, textual_query, text_symbolic_query_idx, query_structure
    
    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count
    
class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


class TestDataset(Dataset):
    def __init__(self, queries, nprod, nentity, nrelation, dataset, text_symbolic_queries2id):
        # queries is a list of (query, query_structure) pairs
        self.len = len(queries)
        self.queries = queries
        self.nprod = nprod
        self.nentity = nentity
        self.nrelation = nrelation
        self.dataset = dataset 
        self.text_symbolic_queries2id = text_symbolic_queries2id

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure = self.queries[idx][1]
        if self.dataset == 'prime':
            negative_sample = torch.LongTensor(range(self.nentity))
        elif self.dataset == 'amazon':
            negative_sample = torch.LongTensor(range(self.nprod))
        return negative_sample, query, self.text_symbolic_queries2id[query[0]], flatten(query[0]), query[1], query_structure
    
    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        unflatten_query = [_[1] for _ in data]
        text_symbolic_query_idx = [_[2] for _ in data]
        symbolic_query = [_[3] for _ in data]
        textual_query = [_[4] for _ in data]
        query_structure = [_[5] for _ in data]
        return negative_sample, unflatten_query, text_symbolic_query_idx, symbolic_query, textual_query, query_structure
    