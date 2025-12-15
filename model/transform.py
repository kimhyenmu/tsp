import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    def __init__(self, d_model=128, num_layers=4, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.time_embedding = nn.Embedding(24, d_model // 4)
        self.minute_embedding = nn.Embedding(60, d_model // 8)
        self.dow_embedding = nn.Embedding(7, d_model // 8)
        self.traffic_proj = nn.Linear(24, d_model // 2)
        
        # 🔥 Fusion layer with activation
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 🔥 GELU 활성화 함수 사용 (Dead Neuron 방지)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # ReLU 대신 GELU
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, gnn_features, start_hour, start_minute, day_of_week, traffic_profile):
        batch_size, num_nodes, _ = gnn_features.shape
        
        time_emb = self.time_embedding(start_hour)
        minute_emb = self.minute_embedding(start_minute)
        dow_emb = self.dow_embedding(day_of_week)
        traffic_emb = self.traffic_proj(traffic_profile)
        
        context = torch.cat([time_emb, minute_emb, dow_emb, traffic_emb], dim=-1)
        context = context.unsqueeze(1).expand(-1, num_nodes, -1)
        
        x = torch.cat([gnn_features, context], dim=-1)
        x = self.fusion_layer(x)
        # Positional Encoding 제거 - TSP에서는 입력 노드에 순서가 없어야 함!
        # x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.norm(x)
        
        return x