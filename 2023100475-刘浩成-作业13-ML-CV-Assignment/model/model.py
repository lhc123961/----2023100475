import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """给时间序列加上位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, time, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=132, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=512,
                 num_classes=6, dropout=0.1):
        super().__init__()
        # 把132维投影到d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: (B, T, 132)
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)         # (B, T, d_model)
        x = x.mean(dim=1)               # 对时间求平均 → (B, d_model)
        x = self.dropout(x)
        x = self.classifier(x)          # (B, num_classes)
        return x