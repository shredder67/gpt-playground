import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Single-head self-attention block"""
    def __init__(self, head_size, block_size, emb_size):
        super().__init__()

        self.head_size = head_size
        self.key = nn.Linear(emb_size, head_size, bias=False) # what i look for
        self.query = nn.Linear(emb_size, head_size, bias=False) # what i have
        self.value = nn.Linear(emb_size, head_size, bias=False) # what i will return

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
        out = wei @ v # BxTxT @ BxTxhead_size => BxTxhead_size
        # when head is single, head_size is equal to emb_size, so shape is same to input
        
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, block_size, emb_size, n_heads):
        super().__init__()
        self.head_size = head_size
        self.n_heads = n_heads
        self.heads = nn.ModuleList([SelfAttention(head_size, block_size, emb_size) for _ in range(n_heads)])

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        out = torch.cat(heads, dim=-1)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        ])

    def forward(self, x):
        return self.net(x)


