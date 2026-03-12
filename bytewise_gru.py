import torch
import torch.nn as nn

class BytewiseGRU(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=1, dropout=0.0):
        super(BytewiseGRU, self).__init__()
        self.vocab_size = 256
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, h = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, h
