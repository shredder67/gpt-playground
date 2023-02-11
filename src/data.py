from collections.abc import Iterable, Iterator
from typing import Tuple

import torch

class TextDataset:
    """Text dataset abstraction over text data fully in-memory"""
    def __init__(self, data, **kwargs):
        if isinstance(data, TextDataset): # case when creating a subset of dataset by slicing
            self.vocab = data.vocab
            self.vocab_size = data.vocab_size
            self.stoi = data.stoi
            self.itos = data.itos
            self.data = data.data[kwargs['idx']]
            return
        text = data
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

        self.data = torch.tensor(self._encode(text), dtype=torch.long)

    def _encode(self, text):
        return [self.stoi[s] for s in text]
    
    def _decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
    
    def train_test_split(self,train_size=0.8):
        train_indices = slice(int(len(self.data) * train_size))
        test_indices = slice(int(len(self.data) * train_size), -1)
        return TextDataset(self, idx=train_indices), TextDataset(self, idx=test_indices)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f"TextDataset with {self.vocab_size} unique characters and {len(self.data)} characters in total."


class BlockReader(Iterable):
    """Block sized data loader for auto-regression task."""
    def __init__(self, text_dataset, block_size=8, batch_size=4, length_before_new_iter=1000, device='cpu'):
        self.text_dataset = text_dataset
        self.block_size = block_size
        self.batch_size = batch_size
        self.length_before_new_iter = length_before_new_iter
        self.device = device

    def __iter__(self):
        return BlockReaderIterator(self)


class BlockReaderIterator(Iterator):
    def __init__(self, reader: BlockReader):
        self.reader = reader
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cur > self.reader.length_before_new_iter:
            raise StopIteration
        self.cur += 1
        ix = torch.randint(len(self.reader.text_dataset) - self.reader.block_size, size=(self.reader.batch_size,))
        x = torch.stack([self.reader.text_dataset[i : i + self.reader.block_size] for i in ix]).to(self.reader.device)
        y = torch.stack([self.reader.text_dataset[i + 1: i + self.reader.block_size + 1] for i in ix]).to(self.reader.device)
        return x, y


def read_text_data(datapath):
    with open(datapath, mode='r', encoding='utf-8') as f:
        text = f.read()
    return text


def preprocess_data(datapath='./data/input.txt') -> Tuple[TextDataset, TextDataset]:
    raw_text = read_text_data(datapath)
    text_dataset = TextDataset(raw_text)
    train_ds, test_ds = text_dataset.train_test_split(0.9)
    # the reason it is done this way is that to make sure that both train and test datasets have the same vocab
    # otherwise, the test vocab will be different and the quality of the model will be lower

    return train_ds, test_ds
