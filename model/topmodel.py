import torch
import torch.nn as nn
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
        
        # GNN (패딩 노드 제외하고 처리)
        gnn_outputs = []
        for i in range(batch_size):
            n = num_nodes_list[i]
            coords = node_features[i, :n, :]  # 유효 노드만!
            
            # k-NN 그래프 (k가 노드 수보다 클 수 없음)
            k = min(self.k_neighbors, n - 1)
            edge_index = build_knn_graph_manual(coords, k=k)
            
            gnn_out = self.gnn_encoder(coords, edge_index)
            
            # 패딩 추가 (max_nodes 크기로 맞추기)
            if n < max_nodes:
                padding = torch.zeros(max_nodes - n, self.hidden_dim, device=device)
                gnn_out = torch.cat([gnn_out, padding], dim=0)
            
            gnn_outputs.append(gnn_out)
        
        gnn_features = torch.stack(gnn_outputs)
        
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
        
        routes, logits, predicted_times, attention_weights = self.pointer_decoder(
            encoder_output=context_features,
            node_coords=node_features,
            start_hour=batch['start_hour'],
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