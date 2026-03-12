import torch
import torch.nn as nn

class BytewiseGRU(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=64, hidden_dim=128, num_layers=1, dropout=0.0):
        super(BytewiseGRU, self).__init__()
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_dim)
        self.embed = self.embedding
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, h = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, h

    def init_stream(self, max_len: int, batch_size: int, device, dtype: torch.dtype):
        device = torch.device(device)
        if dtype not in (torch.float16, torch.float32, torch.float64):
            dtype = torch.float32

        h0 = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size,
                         device=device, dtype=dtype)

        model = self

        class _Stream:
            def __init__(self, model, h):
                self.model = model
                self.h = h

            @torch.no_grad()
            def step(self, tokens):
                # tensor shape (batch,)
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(1)  # -> (batch, 1)
                tokens = tokens.to(self.h.device)
                logits, self.h = self.model(tokens, hidden=self.h)
                return logits.squeeze(1)

        return _Stream(model, h0)

