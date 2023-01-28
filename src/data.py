from collections.abc import Iterable, Iterator
from typing import Tuple

import torch

class TextDataset:
    """Text dataset abstraction over text data fully in-memory"""
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

        self.stoi = {s:i for i,s in enumerate(self.vocab)}
        self.itos = {i:s for s,i in self.stoi.items()}

        self.data = torch.tensor(self._encode(text), dtype=torch.long)

    def _encode(self, text):
        return [self.stoi[s] for s in text]
    
    def _decode(self, indices):
        return ''.join([self.itos[i] for i in indices])
    
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        return self.data[idx]


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

    n = int(0.9 * len(raw_text))
    train_text = raw_text[:n]
    test_text = raw_text[n:]

    train_ds = TextDataset(train_text)
    test_ds = TextDataset(test_text)

    return train_ds, test_ds
