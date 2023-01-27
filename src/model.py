import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # logits: b*t => b*t*c
        logits =  self.embedding(x)        

        if targets is not None:
            # cross-entropy for predicted distribution
            # (in this case, embedding layer actually represents trainable statistics
            # for next character prediction, i.e. each row is log p(x_i+1 | x_i))
            # and targets is essentially array of next tokens per each
            b, t, c = logits.shape
            logits = logits.view(b*t, c)
            targets = torch.flatten(targets)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # last logit in each batch
            logits = logits[:, -1, :]
            # turn logits into probability
            probs = F.softmax(logits, dim=1)
            # sample next token idx per batch
            idx_next = torch.multinomial(probs, num_samples=1)
            # concatenate predicted idx to sequence, repeat
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

  
@torch.no_grad()
def eval_train_test_loss(model, train_br, test_br):
    # count of evaluation steps is equal to length of test iter
    eval_steps = test_br.length_before_new_iter 
    performance = {}
    model.eval()
    train_br.disable_state_update()
    for split in ['train', 'test']:
        losses = torch.zeros((eval_steps,))
        br_iter = iter(train_br) if split=='train' else iter(test_br)
        for i in range(eval_steps): # expected to be less then TRAIN_EPOCH_NUM_STEPS
            xb, yb = next(br_iter)
            losses[i] = model(xb, yb)[1].item()
        performance[split] = torch.mean(losses)
    model.train()
    train_br.enable_state_update()
    return performance


def train_lm(model, train_blockreader, test_blockreader, learning_rate=1e-3, eval_every_iter=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_blockreader):
        if i % eval_every_iter == 0:
            losses = eval_train_test_loss(model, train_blockreader, test_blockreader)
            print(f"step #{i} train_loss: {losses['train']:.4f}, test_loss: {losses['test']:.4f}")
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

    
