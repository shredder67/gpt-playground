import torch
import torch.nn as nn
import torch.nn.functional as F

### Toy section, code samples for intuition behind attention mechanism (not used anywhere) ###

def cbow_average(x: torch.Tensor) -> torch.Tensor:
    """toy function, illustrating average masking of all previous tokens in sequence"""
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T)) # lower triangular matrix of ones
    wei /= wei.sum(dim=1, keepdim=True) # normalize row-wise
    xbow = wei @ x # B(broadcasted)xTxT @ BxTxC => BxTxC
    return xbow


def softmax_cbow_average(x: torch.Tensor) -> torch.Tensor:
    """toy function, average masking + softmax"""
    B, T, C = x.shape
    tril = torch.tril(torch.ones(T, T)) # lower triangular matrix of ones
    wei = torch.zeros(size=(T, T))
    wei = wei.masked_fill(tril==0, float('-inf'))
    # apllies normalize(exp()) row-wise (in last dim)
    # since exp(-inf) -> 0 and exp(0) = 1, this is equivalent to previous version of function
    # the difference is that now wei can be used as a weight matrix for reweighting x previous T features
    # and, in particular, will be able to store information about T affinities 
    wei = F.softmax(wei, dim=-1) 
    xbow = wei @ x
    return xbow

### End of toy section ###

# TODO: finish this
class SelfAttentionBlock(nn.Module):
    """Single-head self-attention block"""
    def __init__(self, head_size, block_size, emb_size):
        self.head_size = head_size
        self.key = nn.Linear(emb_size, head_size, bias=False) # what i look for
        self.query = nn.Linear(emb_size, head_size, bias=False) # what i have
        self.value = nn.Linear(emb_size, head_size, bias=False) # what i will return

        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, mask):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.tranpose(-2, -1) # BxTxhead_size @ Bxhead_sizexT => BxTxT
        wei /= self.head_size ** 0.5 # scale by sqrt(d_k), to make sure that softmax doesn't blow up
        wei = wei.masked_fill(mask==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # BxTxT @ BxTxhead_size => BxTxhead_size
        
        return out


class BigramLMwithAttention(nn.Module):
    def __init__(self, vocab_size, emb_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(block_size, emb_size)

        self.sa_head = SelfAttentionBlock(emb_size, emb_size) # at the moment, head_size == emb_size

        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, x, targets=None):
        # x -> embeddings: b*t => b*t*c
        embeddings =  self.embedding(x)
        # positional encodings - t*c
        pos_embeddings = self.pos_embedding(torch.arange(self.block_size, device=x.device))

        x = embeddings + pos_embeddings
        x = self.sa_head(x)
        logits = self.lm_head(x)  

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