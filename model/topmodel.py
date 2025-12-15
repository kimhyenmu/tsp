import torch
import torch.nn as nn
import torch.nn.functional as F  # 🔥 추가
from torch_geometric.nn import knn_graph
from .gnn import GraphEncoder
from .transform import ContextEncoder
from .pointer import PointerDecoder
from .gnn import build_knn_graph_manual, build_fully_connected_graph 
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
        k_neighbors=10,
        bypass_gnn=False  # 🔥 GNN Bypass 플래그 추가
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.bypass_gnn = bypass_gnn  # 🔥 저장
        
        # 🔥 단순 Linear Embedding (GNN Bypass 시 사용)
        # bias=False로 편향이 차이를 덮는 것 방지!
        self.simple_embedding = nn.Linear(node_dim, hidden_dim, bias=False)
        
        # 🔥 가중치를 크게 초기화 (gain=10)
        nn.init.xavier_uniform_(self.simple_embedding.weight, gain=10.0)
        
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
        
    def forward(self, batch, training=True, teacher_forcing_ratio=0.5, debug=False):
        node_features = batch['node_features']
        batch_size = node_features.size(0)
        max_nodes = node_features.size(1)
        device = node_features.device
        num_nodes_list = batch['num_nodes']
        
        # ============================================================
        # 🔥 GNN Bypass 모드: 단순 Linear Embedding만 사용
        # ============================================================
        if self.bypass_gnn:
            # 🔥 좌표를 직접 Linear로 임베딩 (GNN 없이, bias 없음!)
            gnn_features = self.simple_embedding(node_features)  # (batch, max_nodes, hidden)
            # LayerNorm이나 활성화 함수 없이 순수하게 Linear만!
            
            if debug:
                print("\n" + "="*60)
                print("🔍 [DEBUG] GNN BYPASS 모드 - 단순 Linear Embedding 사용")
                print("="*60)
                print(f"입력 좌표 (node_features) shape: {node_features.shape}")
                print(f"임베딩 출력 shape: {gnn_features.shape}")
                
                # 🔥 핵심: 각 노드의 임베딩이 다른지 확인
                print("\n📊 노드별 임베딩 값 (첫 번째 샘플):")
                n = num_nodes_list[0]
                for i in range(min(5, n)):
                    emb = gnn_features[0, i, :5].detach().cpu().numpy()
                    coord = node_features[0, i].detach().cpu().numpy()
                    print(f"   노드 {i}: 좌표={coord}, 임베딩={emb}...")
                
                # 노드 간 임베딩 차이 확인
                if n >= 2:
                    diff = (gnn_features[0, 0] - gnn_features[0, 1]).abs().mean().item()
                    print(f"\n🔥 노드0 vs 노드1 임베딩 차이: {diff:.6f}")
                    if diff < 1e-5:
                        print("   ❌ 경고: 임베딩이 거의 동일함!")
                    else:
                        print("   ✅ 임베딩이 서로 다름 (정상)")
                print("="*60 + "\n")
        else:
            # ============================================================
            # 🔥 원래 GNN 처리 (완전 연결 그래프 사용)
            # ============================================================
            gnn_outputs = []
            for i in range(batch_size):
                n = num_nodes_list[i]
                coords = node_features[i, :n, :]
                edge_index = build_fully_connected_graph(coords)
                gnn_out = self.gnn_encoder(coords, edge_index)
                
                if n < max_nodes:
                    pad_size = max_nodes - n
                    gnn_out = F.pad(gnn_out, (0, 0, 0, pad_size), mode='constant', value=0)
                
                gnn_outputs.append(gnn_out)
            
            gnn_features = torch.stack(gnn_outputs, dim=0)
            
            if debug:
                print("\n" + "="*60)
                print("🔍 [DEBUG] GNN 모드 - GraphEncoder 사용")
                print("="*60)
                n = num_nodes_list[0]
                print(f"GNN 출력 shape: {gnn_features.shape}")
                for i in range(min(5, n)):
                    emb = gnn_features[0, i, :5].detach().cpu().numpy()
                    coord = node_features[0, i].detach().cpu().numpy()
                    print(f"   노드 {i}: 좌표={coord}, 임베딩={emb}...")
                
                if n >= 2:
                    diff = (gnn_features[0, 0] - gnn_features[0, 1]).abs().mean().item()
                    print(f"\n🔥 노드0 vs 노드1 임베딩 차이: {diff:.6f}")
                print("="*60 + "\n")
        
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