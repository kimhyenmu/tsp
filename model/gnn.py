import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import knn_graph


def build_fully_connected_graph(node_features):
    """
    🔥 완전 연결 그래프 생성 (TSP/VRP에 적합)
    모든 노드가 서로 연결됨 (self-loop 제외)
    
    Args:
        node_features: (N, 2) 좌표
    
    Returns:
        edge_index: (2, N*(N-1)) - 모든 쌍의 연결
    """
    N = node_features.size(0)
    device = node_features.device
    
    # 모든 노드 쌍 생성 (self-loop 제외)
    # source: [0,0,0,...,1,1,1,...,N-1,N-1,...]
    # target: [1,2,3,...,0,2,3,...,0,1,2,...]
    source = torch.arange(N, device=device).unsqueeze(1).expand(N, N).reshape(-1)
    target = torch.arange(N, device=device).unsqueeze(0).expand(N, N).reshape(-1)
    
    # Self-loop 제거 (i != j인 경우만)
    mask = source != target
    source = source[mask]
    target = target[mask]
    
    edge_index = torch.stack([source, target], dim=0)  # (2, N*(N-1))
    
    return edge_index


def build_knn_graph_manual(node_features, k=10):
    """
    🔥 개선된 k-NN 그래프 생성
    k가 노드 수보다 크면 완전 연결 그래프로 fallback
    """
    N = node_features.size(0)
    
    # 🔥 노드 수가 적으면 완전 연결 그래프 사용
    if N <= k + 1:
        return build_fully_connected_graph(node_features)
    
    with torch.no_grad():
        coords = node_features.detach()
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        
        # Self-loop 제외
        mask = torch.eye(N, device=node_features.device, dtype=torch.bool)
        dist = dist.masked_fill(mask, float('inf'))
        
        # k개의 가장 가까운 이웃
        _, indices = torch.topk(dist, k, dim=1, largest=False)
        
        source = torch.arange(N, device=node_features.device).unsqueeze(1).expand(-1, k).reshape(-1)
        target = indices.reshape(-1)
        
        edge_index = torch.stack([source, target], dim=0)
    
    return edge_index


class GraphEncoder(nn.Module):
    """
    🔥 개선된 Graph Attention Network
    - 완전 연결 그래프 지원
    - 강력한 Residual Connection (원본 좌표 정보 보존)
    - 초기 학습에서도 안정적으로 작동
    """
    def __init__(self, node_dim=2, hidden_dim=128, num_layers=3, num_heads=4, 
                 dropout=0.1, use_distance_weight=True):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 🔥 입력 임베딩 (좌표 → hidden_dim)
        # LeakyReLU로 Dead Neuron 방지
        self.input_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
        
        # 🔥 원본 좌표를 직접 보존하는 별도 projection
        self.coord_proj = nn.Linear(node_dim, hidden_dim)
        
        # GAT 레이어들
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            layer = GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True  # Self-loop 포함
            )
            self.gat_layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # FFN (LeakyReLU로 Dead Neuron 방지)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # 🔥 최종 출력: GNN 출력 + 원본 좌표 정보 결합
        self.output_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: (N, node_dim) - 노드 좌표
            edge_index: (2, E) - 엣지 연결 정보
        
        Returns:
            output: (N, hidden_dim) - 노드 임베딩
        """
        # 🔥 원본 좌표 정보 보존 (Residual Path)
        coord_embedding = self.coord_proj(node_features)  # (N, hidden_dim)
        
        # 입력 임베딩
        x = self.input_embedding(node_features)  # (N, hidden_dim)
        
        # 🔥 초기 임베딩 저장 (전체 Residual용)
        initial_embedding = x
        
        # GAT 레이어 통과
        for gat, norm in zip(self.gat_layers, self.norms):
            # GAT 연산
            gat_out = gat(x, edge_index)
            
            # 🔥 Pre-LayerNorm + Residual + LeakyReLU
            x = norm(x + gat_out)
            x = F.leaky_relu(x, negative_slope=0.1)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        
        # 🔥 핵심: 원본 좌표 정보와 GNN 출력을 결합
        # 이렇게 하면 GNN이 학습 초기에 엉망이어도 좌표 정보는 보존됨
        combined = torch.cat([x, coord_embedding], dim=-1)  # (N, hidden_dim * 2)
        output = self.output_fusion(combined)  # (N, hidden_dim)
        output = self.output_norm(output + initial_embedding)  # 🔥 전체 Residual
        
        return output