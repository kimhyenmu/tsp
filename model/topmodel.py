import torch
import torch.nn as nn
import torch.nn.functional as F  # 🔥 추가
from torch_geometric.nn import knn_graph
from .gnn import GraphEncoder
from .transform import ContextEncoder
from .pointer import PointerDecoder
from .gnn import build_knn_graph_manual 
class HybridRoutingModel(nn.Module):
    def __init__(
        self,
        node_dim=2,
        hidden_dim=128,
        gnn_layers=3,
        gnn_heads=4,
        tf_layers=4,
        tf_heads=8,
        tf_ff_dim=512,
        pointer_heads=4,
        dropout=0.1,
        k_neighbors=10
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        self.gnn_encoder = GraphEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=gnn_heads,
            dropout=dropout
        )
        
        self.context_encoder = ContextEncoder(
            d_model=hidden_dim,
            num_layers=tf_layers,
            num_heads=tf_heads,
            d_ff=tf_ff_dim,
            dropout=dropout
        )
        
        self.pointer_decoder = PointerDecoder(
            hidden_dim=hidden_dim,
            num_heads=pointer_heads,
            dropout=dropout
        )
        
    def forward(self, batch, training=True, teacher_forcing_ratio=0.5):
        node_features = batch['node_features']
        batch_size = node_features.size(0)
        max_nodes = node_features.size(1)
        device = node_features.device
        num_nodes_list = batch['num_nodes']
        
        # 🔥 GNN 처리 (Gradient 흐름 유지 확인)
        gnn_outputs = []
        for i in range(batch_size):
            n = num_nodes_list[i]
            
            # 🔥 유효 노드만 추출 (slicing은 gradient 유지)
            coords = node_features[i, :n, :]
            
            # k-NN 그래프 생성 (이 부분은 gradient 불필요)
            k = min(self.k_neighbors, n - 1)
            edge_index = build_knn_graph_manual(coords, k=k)
            
            # 🔥 GNN 인코딩 (이 부분에서 gradient 필요!)
            gnn_out = self.gnn_encoder(coords, edge_index)
            
            # 🔥 패딩 추가 시 gradient 유지를 위해 F.pad 사용
            if n < max_nodes:
                # (max_nodes - n, hidden_dim) 크기의 패딩
                # F.pad는 gradient를 유지함
                pad_size = max_nodes - n
                gnn_out = F.pad(gnn_out, (0, 0, 0, pad_size), mode='constant', value=0)
            
            gnn_outputs.append(gnn_out)
        
        # 🔥 stack은 gradient를 유지
        gnn_features = torch.stack(gnn_outputs, dim=0)
        
        # Transformer
        context_features = self.context_encoder(
            gnn_features,
            batch['start_hour'],
            batch['start_minute'],
            batch['day_of_week'],
            batch['traffic_profile']
        )
        
        # Pointer
        teacher_route = batch.get('actual_route', None) if training else None
        
        # 🔥 num_nodes_list를 전달하여 패딩 마스킹 적용
        routes, logits, predicted_times, attention_weights = self.pointer_decoder(
            encoder_output=context_features,
            node_coords=node_features,
            start_hour=batch['start_hour'],
            num_nodes_list=num_nodes_list,  # 🔥 추가
            teacher_route=teacher_route,
            teacher_forcing_ratio=teacher_forcing_ratio,
            training=training
        )
        
        return {
            'routes': routes,
            'logits': logits,
            'predicted_times': predicted_times,
            'attention_weights': attention_weights
        }