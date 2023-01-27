from src import *

import random
from typing import Callable

import torch

DATA_PATH = './data/input.txt'
BATCH_SIZE = 32
TRAIN_EPOCH_NUM_STEPS = 1000
TEST_EPOCH_NUM_STEPS = 200
LEARNING_RATE = 1e-2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RAND_SEED = 42
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

def generate_sample_text(model: BigramLanguageModel, text_length: int, decode_f: Callable) -> str:
    idx = torch.zeros((1, 1), dtype=torch.long)
    predicted_seq = model.generate(idx, max_new_tokens=text_length)[0].tolist()
    text = decode_f(predicted_seq)
    return text

# TODO: fix train/eval loop of  bigram lm 

def main():
    train_ds, test_ds = preprocess_data(DATA_PATH)
    
    train_block_reader = BlockReader(
        train_ds, 
        batch_size=BATCH_SIZE,
        length_before_new_iter=TRAIN_EPOCH_NUM_STEPS,
        device=DEVICE
    )
    test_block_reader = BlockReader(
        test_ds, 
        batch_size=BATCH_SIZE,
        length_before_new_iter=TEST_EPOCH_NUM_STEPS,
        device=DEVICE
    )

    m = BigramLanguageModel(train_ds.vocab_size).to(DEVICE)
    
    print(train_lm(m, train_block_reader, test_block_reader, LEARNING_RATE))

    m.eval()
    print(generate_sample_text(m, text_length=20, decode_f=train_ds._decode))

if __name__ == "__main__":
    main()