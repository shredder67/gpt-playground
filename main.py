from src import *

import random
from typing import Callable

import torch

DATA_PATH = './data/input.txt'
BATCH_SIZE = 32
TRAIN_EPOCH_NUM_STEPS = 10_000
TEST_EPOCH_NUM_STEPS = 500
LEARNING_RATE = 1e-2
EMBEDDING_SIZE = 32
BLOCK_SIZE = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RAND_SEED = 42
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)


def generate_sample_text(model: BigramLanguageModel, text_length: int, decode_f: Callable) -> str:
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    predicted_seq = model.generate(idx, max_new_tokens=text_length)[0].tolist()
    text = decode_f(predicted_seq)
    return text


def main():
    train_ds, test_ds = preprocess_data(DATA_PATH)
    
    train_block_reader = BlockReader(
        train_ds,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        length_before_new_iter=TRAIN_EPOCH_NUM_STEPS,
        device=DEVICE
    )
    test_block_reader = BlockReader(
        test_ds,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        length_before_new_iter=TEST_EPOCH_NUM_STEPS,
        device=DEVICE
    )

    # m = BigramLanguageModel(train_ds.vocab_size).to(DEVICE)
    # train_lm(m, train_block_reader, test_block_reader, LEARNING_RATE)
    # print(generate_sample_text(m, text_length=20, decode_f=train_ds._decode))

    m = BigramLMwithAttention(train_ds.vocab_size, EMBEDDING_SIZE, BLOCK_SIZE).to(DEVICE)

if __name__ == "__main__":
    main()