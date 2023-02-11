from src import *

import os
import random
import argparse
from typing import Callable

import torch

DATA_PATH = './data/input.txt'

### Hyperparameters ###
BATCH_SIZE = 64
TRAIN_EPOCH_NUM_STEPS = 5000
TEST_EPOCH_NUM_STEPS = 200
LEARNING_RATE = 3e-4
EMBEDDING_SIZE = 300
BLOCK_SIZE = 400
N_LAYER = 6
NUM_HEADS = 6
DROP_PROB = 0.2
#######################

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

RAND_SEED = 42
random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--save_model_as', 
    type=str, 
    default=None, 
    help='saves model by name to /models folder'
)
argparser.add_argument(
    '--test_inference',
    type=str, 
    default=None,
    help='tests inference of model loaded from /models folder by name'
)


def check_args(args):
    if args.save_model_as is not None:
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if '.' in args.save_model_as:
            _, ext = args.split_model.split('.')
            if ext != 'pt':
                raise ValueError('model should be saved as .pt or .pth file')
        else:
            args.save_model_as += '.pt'
        if os.path.exists('./models/' + args.save_model_as):
            raise ValueError(f'saved model with same name as "{args.save_model_as}" alreadery exists')
    if args.test_inference is not None:
        if not os.path.exists('./models') \
            or not os.path.exists('./models/' + args.test_inference):
            raise ValueError(f'model with name "{args.test_inference}" does not exist in /models folder')


def generate_sample_text(model: nn.Module, text_length: int, decode_f: Callable, text_seed: str=None, encode_f: Callable=None) -> str:
    model.eval()
    if text_seed is not None:
        idx = torch.tensor(encode_f(text_seed), dtype=torch.long)[None, :].to(DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long).to(DEVICE)
    predicted_seq = model.generate(idx, max_new_tokens=text_length)[0].tolist()
    text = decode_f(predicted_seq)
    return text


def test_inference(model: nn.Module, text_lengths, seed_texts, encode_f: Callable, decode_f: Callable) -> None:
    for l, s in zip(text_lengths, seed_texts):
        print(f"Input seed: {s}")
        print(f"Generated text:\n{generate_sample_text(model, l, decode_f, s, encode_f)}\n")


def main():
    args = argparser.parse_args()
    check_args(args)

    train_ds, test_ds = preprocess_data(DATA_PATH)

    if args.test_inference is not None:
        m = BigramTransformer(train_ds.vocab_size, EMBEDDING_SIZE, BLOCK_SIZE, DROP_PROB, N_LAYER, NUM_HEADS).to(DEVICE)
        m.load_state_dict(torch.load(f'./models/{args.test_inference}'))
        test_inference(m, [1000], ['How dare you!'], train_ds._encode, train_ds._decode)
        return
    
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

    # check taht rain_ds and test_ds vocab and stoi, itos are the same
    assert train_ds.vocab_size == test_ds.vocab_size
    assert train_ds.stoi == test_ds.stoi
    assert train_ds.itos == test_ds.itos

    
    m = BigramTransformer(train_ds.vocab_size, EMBEDDING_SIZE, BLOCK_SIZE, DROP_PROB, N_LAYER, NUM_HEADS).to(DEVICE)
    train_lm(m, train_block_reader, test_block_reader, LEARNING_RATE)
    
    if args.save_model_as is not None:
        torch.save(m.state_dict(), f'./models/{args.save_model_as}')

    print(generate_sample_text(m, text_length=100, decode_f=train_ds._decode))

if __name__ == "__main__":
    main()