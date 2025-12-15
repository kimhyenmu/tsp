import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from .gnn import GraphEncoder
from .transform import ContextEncoder
from .pointer import PointerDecoder, log_tensor_stats
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
        
        # 🔥🔥🔥 [최종 처방] ID Embedding 사용!
        # 좌표(continuous) 대신 노드 ID(discrete)로 임베딩
        MAX_NODES = 200  # 최대 노드 수
        self.node_embedding = nn.Embedding(MAX_NODES, hidden_dim)
        
        # 🔥 안전한 초기화 (작은 값으로 시작)
        nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.1)
        
        # 좌표 정보도 추가로 사용
        self.coord_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.coord_proj.weight, gain=0.1)  # 🔥 작은 gain
        
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
            # 🔥🔥🔥 [최종 처방] ID Embedding 사용!
            
            # 노드 ID 생성
            node_ids = torch.arange(max_nodes, device=device)
            node_ids = node_ids.unsqueeze(0).expand(batch_size, -1)
            
            # 🔥🔥🔥 ID Embedding × 10 (Signal Boosting!)
            id_emb = self.node_embedding(node_ids) * 10.0
            
            # 🔥 좌표 임베딩도 10배
            coord_emb = self.coord_proj(node_features) * 10.0
            
            # 결합
            gnn_features = id_emb + coord_emb
            
            # ============================================================
            # 🔬 CT Scan Point (1): Embedding 직후
            # ============================================================
            if debug:
                print("\n" + "="*60)
                print("🔬 CT SCAN Point (1): Embedding 직후")
                print("="*60)
                log_tensor_stats("입력 좌표 (node_features)", node_features, indent=1)
                log_tensor_stats("ID Embedding", id_emb, indent=1)
                log_tensor_stats("Coord Embedding", coord_emb, indent=1)
                log_tensor_stats("결합 (id + coord)", gnn_features, indent=1)
                
                # 노드 간 차이 확인
                n = num_nodes_list[0]
                if n >= 2:
                    diff = (gnn_features[0, 0] - gnn_features[0, 1]).abs().mean().item()
                    print(f"   🔥 노드0 vs 노드1 차이: {diff:.6f}")
                    if diff < 0.01:
                        print("   ❌ 차이가 너무 작음!")
                    else:
                        print("   ✅ 차이가 충분함!")
                print("="*60)
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
        
        # ============================================================
        # 🔬 CT Scan Point (2): Encoder 출력 (Decoder 입력 전)
        # ============================================================
        if debug:
            print("\n" + "="*60)
            print("🔬 CT SCAN Point (2): Encoder 출력")
            print("="*60)
            log_tensor_stats("Context Features (encoder out)", context_features, indent=1)
            
            # 노드 간 차이 확인
            n = num_nodes_list[0]
            if n >= 2:
                diff = (context_features[0, 0] - context_features[0, 1]).abs().mean().item()
                print(f"   🔥 노드0 vs 노드1 차이: {diff:.6f}")
            print("="*60)
        
        # Pointer
        teacher_route = batch.get('actual_route', None) if training else None
        
        # 🔥 num_nodes_list를 전달하여 패딩 마스킹 적용
        routes, logits, predicted_times, attention_weights = self.pointer_decoder(
            encoder_output=context_features,
            node_coords=node_features,
            start_hour=batch['start_hour'],
            num_nodes_list=num_nodes_list,
            teacher_route=teacher_route,
            teacher_forcing_ratio=teacher_forcing_ratio,
            training=training,
            debug=debug  # 🔥 debug 전달
        )
        
        return {
            'routes': routes,
            'logits': logits,
            'predicted_times': predicted_times,
            'attention_weights': attention_weights
        }