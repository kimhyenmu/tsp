import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import knn_graph


def build_knn_graph_manual(node_features, k=10):
    """
    torch-cluster 없이 k-NN 그래프 생성
    
    Args:
        node_features: (N, 2) 좌표
        k: 이웃 개수
    
    Returns:
        edge_index: (2, E)
    """
    N = node_features.size(0)
    
    # 거리 행렬 계산
    # (N, 1, 2) - (1, N, 2) = (N, N, 2)
    diff = node_features.unsqueeze(1) - node_features.unsqueeze(0)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    
    # 자기 자신과의 거리를 무한대로
    dist.fill_diagonal_(float('inf'))
    
    # 각 노드별 k개의 가까운 이웃 찾기
    _, indices = torch.topk(dist, k, dim=1, largest=False)  # (N, k)
    
    # edge_index 구성
    source = torch.arange(N, device=node_features.device).unsqueeze(1).expand(-1, k).reshape(-1)
    target = indices.reshape(-1)
    
    edge_index = torch.stack([source, target], dim=0)  # (2, N*k)
    
    return edge_index


class GraphEncoder(nn.Module):
    """Graph Attention Network"""
    def __init__(self, node_dim=2, hidden_dim=128, num_layers=3, num_heads=4, 
                 dropout=0.1, use_distance_weight=True):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.input_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            layer = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True
            )
            
            self.gat_layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_index):
        x = self.input_embedding(node_features)
        
        for gat, norm in zip(self.gat_layers, self.norms):
            out = gat(x, edge_index)
            x = norm(x + out)
            x = F.relu(x)
        
        out = self.ffn(x)
        x = self.ffn_norm(x + out)
        
        return x