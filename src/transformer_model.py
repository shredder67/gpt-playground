import torch
import torch.nn as nn
import torch.nn.functional as F

### Toy section, code samples for intuition behind attention mechanism (not used anywhere) ###

def cbow_average(x: torch.Tensor) -> torch.Tensor:
    """toy function, illustrating average masking of all previous tokens in sequence"""
    B, T, C = x.shape
    wei = torch.tril(torch.ones(T, T)) # lower triangular matrix of ones
    wei /= wei.sum(dim=1, keepdim=True) # normalize row-wise
    xbow = wei @ x # B(broadcasted)xTxT @ BxTxC => BxTxC, equivalent to meaning out previous rows for each row
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

class SelfAttentionBlock(nn.Module):
    """Single-head self-attention block"""
    def __init__(self, head_size, block_size, emb_size, drop_prob):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(emb_size, head_size, bias=False) # what i look for
        self.query = nn.Linear(emb_size, head_size, bias=False) # what i have
        self.value = nn.Linear(emb_size, head_size, bias=False) # what i will return
        self.dropout = nn.Dropout(drop_prob)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        b,t,c = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) # BxTxhead_size @ Bxhead_sizexT => BxTxT
        wei *= self.head_size ** -0.5 # scale by sqrt(d_k), to make sure that softmax doesn't blow up and converge to onehot vector
        wei = wei.masked_fill(self.mask[:t, :t]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # BxTxT @ BxTxhead_size => BxTxhead_size
        # when head is single, head_size is equal to emb_size, so shape is same to input
        
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, block_size, emb_size, n_heads, drop_prob):
        super().__init__()
        self.head_size = head_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([SelfAttentionBlock(head_size, block_size, emb_size, drop_prob) for _ in range(n_heads)])
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out = [head(x) for head in self.heads]
        out = torch.cat(out, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, emb_size, drop_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, block_size, num_heads, drop_prob):
        super().__init__()
        head_size = emb_size // num_heads
        self.sa = MultiHeadAttention(head_size, block_size, emb_size, num_heads, drop_prob)
        self.ff = FeedForward(emb_size, drop_prob)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class BigramTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size, block_size, drop_prob, n_layer=6, num_heads=4):
        super().__init__()
        self.block_size = block_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(block_size, emb_size)

        self.blocks = nn.Sequential(
            *[TransformerBlock(emb_size, block_size, num_heads, drop_prob) for _ in range(n_layer)],
            nn.LayerNorm(emb_size)
        )
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, x, targets=None):
        # x -> embeddings: b*t => b*t*c
        embeddings =  self.embedding(x)
        # positional encodings - t*c
        pos_embeddings = self.pos_embedding(torch.arange(x.shape[1], device=x.device))

        x = embeddings + pos_embeddings
        x = self.blocks(x)
        logits = self.lm_head(x)  # b*t*c => b*t*vocab_size

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
            idx_block = idx[:, -self.block_size:] # truncate to block size
            logits, _ = self(idx_block)
            # last logit in each batch
            logits = logits[:, -1, :]
            # turn logits into probability
            probs = F.softmax(logits, dim=1)
            # sample next token idx per batch
            idx_next = torch.multinomial(probs, num_samples=1)
            # concatenate predicted idx to sequence, repeat
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx