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
