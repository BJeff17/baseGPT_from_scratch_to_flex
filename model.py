import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os

"""

"""




class SlidingWindowDataset(Dataset):
    def __init__(self,  tokenizer, n_context, stride=1, pad_token_id=0, folder_path=None, texts=None):
        self.tokenizer = tokenizer
        self.seq_len = n_context
        self.data = []
        self.stride = stride
        self.pad_token_id = pad_token_id
        if folder_path:
            self.read_from_folder(folder_path, tokenizer, n_context, stride, pad_token_id)
        elif texts:
            for text in texts:
                self.threat_text(text, tokenizer, n_context, stride, pad_token_id)
        else:
            raise ValueError("Either folder_path or texts must be provided.")
        self.vocab_size = len(tokenizer)
        self.n_context = n_context
        
    def read_from_folder(self, folder_path):
        for file in os.listdir(self.folder_path):
            if file.endswith(".txt"):
                with open(os.path.join(self.folder_path, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.threat_text(text, self.tokenizer, self.n_context, self.stride, self.pad_token_id)
    
    
    def threat_text(self, text):
        tokens = self.tokenizer.encode(text)
        n = len(tokens)

        for i in range(0, n, self.stride):
            x = tokens[i:i+self.n_context]
            y = tokens[i+1:i+self.n_context+1]

            if len(x) < self.n_context:
                x = [self.pad_token_id] * (self.n_context - len(x)) + x
            if len(y) < self.n_context:
                y += [self.pad_token_id] * (self.n_context - len(y))

            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class AttentionHead(nn.Module):
    def __init__(self, n_model,n_k, n_v,isMasked=True):
        super().__init__()
        self.scale = math.sqrt(n_k)
        self.Q = nn.Linear(n_model, n_k)
        self.K = nn.Linear(n_model, n_k)
        self.V = nn.Linear(n_model, n_v)
        self.isMasked = isMasked
    def forward(self,x:torch.Tensor):
        Wq,Wk,Wv = self.Q(x),self.K(x),self.V(x)
        attn_weight = (1/self.scale)*Wq@Wk.transpose(-2,-1)

        if self.isMasked:

            T = x.size(1)
            mask = torch.triu(torch.ones(T,T), diagonal=1)
            attn_weight = attn_weight.masked_fill(mask==1, float("-inf"))
        return torch.matmul(F.softmax(attn_weight, dim=-1), Wv)
    
class MultiHead(nn.Module):
    def __init__(self, n_model,n_k, n_v,n_head ):
        super().__init__()
        self.Heads = nn.ModuleList([AttentionHead( n_model,n_k, n_v) for _ in range(n_head) ])
        self.O = nn.Linear(n_head*n_v,n_model)

    def forward(self,x):
        t_cat = torch.cat([h(x) for h in self.Heads], dim=-1)
        output = self.O(t_cat)
        return output

class FeedForward(nn.Module):
    def __init__(self, n_model,n_ff):
        super().__init__()
        self.W1 = nn.Linear(n_model,n_ff)
        self.W2 = nn.Linear(n_ff,n_model)
        self.LNorm = nn.LayerNorm(n_model)

        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        return self.LNorm(x) + x

class Transformer(nn.Module):
    def __init__(self, n_model,n_k, n_v, n_head,n_ff):
        super().__init__() 
        self.AttHeads = MultiHead(n_model,n_k, n_v,n_head)
        self.LNorm = nn.LayerNorm(n_model)
        self.FFF = FeedForward(n_model, n_ff)
    def forward(self, x):
        x += self.AttHeads(x)
        x = self.LNorm(x)
        return self.FFF(x)

class Embedding(nn.Module):
    def __init__(self, vocab_size, n_embed ):
        super().__init__()
        self.Embed = nn.Parameter(torch.randn(vocab_size, n_embed))
    def forward(self, idx):
        return self.Embed[idx]
    
class PositionEncod(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        posEnc = torch.zeros(vocab_size, n_embed)
        positions = torch.arange(0,vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed,2)).float() * (-math.log(10000.0) / n_embed)
        posEnc[:, 0::2] = torch.sin(positions * div_term)   
        posEnc[:, 1::2] = torch.cos(positions * div_term)   
        
        posEnc = posEnc.unsqueeze(0)  
        self.register_buffer('posEnc', posEnc)

    def forward(self, x):
        x = x + self.posEnc[:, :x.size(1)]
        return x 

class B2lM(nn.Module):
    def __init__(self, vocab_size, n_embed, n_block,n_k, n_v, n_head, n_ff):
        super().__init__()
        self.embeds = Embedding(vocab_size, n_embed)
        self.posEmb = PositionEncod(vocab_size, n_embed)
        self.Transformers = nn.Sequential([Transformer(n_embed, n_k, n_v, n_head, n_ff) for _ in n_block])
        self.Out = nn.Linear(n_embed, vocab_size)
    
    def forward(self,x):
        embds = self.embeds(x)
        posEmbd = self.posEmb(x)
        t_out = self.Transformers(embds+posEmbd)
        return self.Out(t_out)
    
    def train(self, datatloader,epochs):
        optim = Adam(self.parameters())
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            loss = 0
            for x,y in datatloader:
                out = self(x)
                out = out.view(-1,out.size(-1))
                y = y.view(-1)
                _loss:torch.Tensor = criterion(out, y)
                optim.zero_grad()
                _loss.backward()
                optim.step()
                loss += _loss.item()
        print(f"epoch {epoch+1} | Loss: {loss/len(datatloader):.4f}")
