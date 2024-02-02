import math

import torch
import torch.nn as nn
import pytorch_lightning as pl

class PositionalEncoding(pl.LightningModule):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: What is the embedding dim?
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PositionalEmbedding(pl.LightningModule):
    def __init__(self, num_frame_patches: int, num_image_patches: int, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1,num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        # TODO: What is n? 
        b, f, n, _ = x.shape
        x = x + self.pos_embedding[:, :f, :n]