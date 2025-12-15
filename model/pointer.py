import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datetime import timedelta  

class PointerAttention(nn.Module):
    """
    🔥 개선된 Pointer Attention
    - Scaled Dot-Product 방식으로 변경 (더 안정적인 gradient)
    - Bahdanau Attention과 Dot-Product Attention 결합
    """
    def __init__(self, hidden_dim, num_heads=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Query, Key, Value projections
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(hidden_dim, hidden_dim)
        
        # 🔥 Attention scale (학습 가능)
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(hidden_dim)))
        
        # 🔥 Temperature (logits 범위 조절)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, query, ref, mask=None):
        """
        Args:
            query: (batch, hidden_dim) - decoder state
            ref: (batch, num_nodes, hidden_dim) - encoder outputs (GNN 출력)
            mask: (batch, num_nodes) - 방문한 노드 마스크
        
        Returns:
            logits: (batch, num_nodes)
            attention_weights: (batch, num_nodes)
        """
        # Query와 Key 변환
        q = self.W_query(query)  # (batch, hidden_dim)
        k = self.W_key(ref)       # (batch, num_nodes, hidden_dim)
        
        # 🔥 Scaled Dot-Product Attention
        # logits = (q @ k^T) / sqrt(d)
        q = q.unsqueeze(1)  # (batch, 1, hidden_dim)
        logits = torch.bmm(q, k.transpose(1, 2))  # (batch, 1, num_nodes)
        logits = logits.squeeze(1)  # (batch, num_nodes)
        
        # 🔥 Scale 적용 (학습 가능한 temperature)
        logits = logits * torch.abs(self.scale) * torch.clamp(self.temperature, min=0.1, max=10.0)
        
        # 마스킹
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), -100.0)
        
        attention_weights = F.softmax(logits, dim=-1)
        
        return logits, attention_weights


class TimePredictor(nn.Module):
    """
    🔥 개선된 시간 예측기
    - 학습 가능한 스케일/바이어스 파라미터 추가
    - 거리 기반 baseline 추정 포함
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.hour_embedding = nn.Embedding(24, hidden_dim // 4)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 🔥 학습 가능한 출력 스케일링 파라미터
        # 데이터 분포에 맞게 자동 조정됨
        self.output_scale = nn.Parameter(torch.tensor(300.0))  # 평균 시간 (초)
        self.output_bias = nn.Parameter(torch.tensor(200.0))   # 최소 시간 (초)
        
    def forward(self, current_node_emb, next_node_emb, current_hour, distance=None):
        batch_size = current_node_emb.size(0)
        
        hour_emb = self.hour_embedding(current_hour)
        
        if distance is None:
            distance = torch.zeros(batch_size, 1, device=current_node_emb.device)
        else:
            distance = distance.unsqueeze(-1)
        
        features = torch.cat([current_node_emb, next_node_emb, hour_emb, distance], dim=-1)
        fused = self.feature_fusion(features)
        raw_output = self.time_head(fused).squeeze(-1)
        
        # 🔥 개선된 출력 변환
        # 1. softplus로 양수 보장
        # 2. 학습 가능한 scale/bias로 데이터 분포에 적응
        # 3. 거리 기반 baseline 추가 (거리가 멀수록 시간 증가)
        base_time = F.softplus(raw_output) * torch.abs(self.output_scale) + torch.abs(self.output_bias)
        
        # 거리 기반 보정 (정규화된 좌표 기준, 거리 * 상수)
        # 거리 0.1 (정규화 좌표) ≈ 실제 몇 km → 시간 보정
        distance_factor = distance.squeeze(-1) * 1000.0  # 거리에 비례한 시간 추가
        
        predicted_time = base_time + distance_factor
        
        # 최소/최대 범위 클램핑 (비정상적인 값 방지)
        predicted_time = torch.clamp(predicted_time, min=60.0, max=7200.0)  # 1분 ~ 2시간
        
        return predicted_time


class PointerDecoder(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.pointer_attention = PointerAttention(hidden_dim, num_heads)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.time_predictor = TimePredictor(hidden_dim, dropout)
        
        # 🔥 Start embedding은 학습 가능하지만, 초기값을 작게 설정
        self.start_embedding = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        
        self.context_summarizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 🔥 Depot(0번 노드) 임베딩을 위한 projection
        self.depot_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, encoder_output, node_coords, start_hour, 
                num_nodes_list=None,  # 🔥 추가: 각 샘플의 유효 노드 수
                teacher_route=None, teacher_forcing_ratio=0.5, training=True):
        batch_size, max_nodes, _ = encoder_output.shape
        device = encoder_output.device
        
        global_context = encoder_output.mean(dim=1)
        global_context = self.context_summarizer(global_context)
        
        routes = []
        logits_sequence = []
        predicted_times = []
        attention_weights_list = []
        
        # LSTM hidden state를 global_context로 초기화
        h0 = global_context.unsqueeze(0).repeat(2, 1, 1)  # (num_layers=2, batch, hidden)
        c0 = torch.zeros_like(h0)
        hidden_state = (h0, c0)
        current_hour = start_hour.clone()
        current_node_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        routes.append(current_node_idx)
        
        # 🔥 Start Token: Depot(0번 노드)의 임베딩 사용
        # 이렇게 하면 GNN 출력이 처음부터 gradient에 포함됨
        depot_emb = encoder_output[:, 0, :]  # 0번 노드(depot)의 임베딩
        current_input = self.depot_proj(depot_emb) + self.start_embedding.expand(batch_size, -1)
        
        # 🔥 핵심 수정: Padding Masking 초기화
        # 각 샘플의 유효 노드 수를 넘어가는 인덱스는 처음부터 마스킹
        mask = torch.zeros(batch_size, max_nodes, device=device)
        mask[:, 0] = 1  # depot 마스킹
        
        # 🔥 Padding 노드 마스킹 (유효 노드 수 이상의 인덱스)
        if num_nodes_list is not None:
            for b in range(batch_size):
                n = num_nodes_list[b]
                if n < max_nodes:
                    mask[b, n:] = 1  # 패딩 노드들을 마스킹
        
        # 🔥 수정: max_nodes가 아닌 가장 긴 시퀀스 기준으로 반복
        # (배치 내 모든 샘플을 처리해야 하므로 max_nodes - 1 사용)
        max_steps = max_nodes - 1
        if num_nodes_list is not None:
            max_steps = max(num_nodes_list) - 1
        
        for step in range(max_steps):
            lstm_out, hidden_state = self.lstm(current_input.unsqueeze(1), hidden_state)
            decoder_state = lstm_out.squeeze(1)
            
            # 🔥 핵심 수정: Loss 계산용과 노드 선택용 logits 분리
            # 1. raw_logits: 마스킹 없음 (Loss 계산용 - 모든 노드에 대한 확률 학습)
            # 2. masked_logits: 마스킹 적용 (노드 선택용 - 중복 방문 방지)
            raw_logits, _ = self.pointer_attention(decoder_state, encoder_output, mask=None)
            masked_logits, attn_weights = self.pointer_attention(decoder_state, encoder_output, mask)
            
            # Loss 계산에는 raw_logits 사용 (Validation에서도 안정적)
            logits_sequence.append(raw_logits)
            attention_weights_list.append(attn_weights)
            
            # 🔥 Teacher Forcing 로직 개선
            # teacher_forcing_ratio=1.0이면 항상 정답 사용
            use_teacher_forcing = (
                training and 
                teacher_route is not None and 
                teacher_forcing_ratio >= 1.0  # 🔥 1.0이면 무조건 적용
            )
            
            if not use_teacher_forcing and training and teacher_route is not None:
                # ratio < 1.0이면 확률적으로 적용
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # 🔥 Teacher Forcing: 정답 경로 사용
                next_node_idx = teacher_route[:, step + 1]
            else:
                # 🔥 Free Running: 모델 예측 사용
                if training:
                    probs = F.softmax(masked_logits / 1.0, dim=-1)
                    next_node_idx = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    next_node_idx = masked_logits.argmax(dim=-1)
            
            routes.append(next_node_idx)
            
            # 🔥 현재 노드와 다음 노드의 임베딩 (GNN 출력에서 가져옴)
            current_node_emb = encoder_output[torch.arange(batch_size), current_node_idx]
            next_node_emb = encoder_output[torch.arange(batch_size), next_node_idx]
            
            current_coords = node_coords[torch.arange(batch_size), current_node_idx]
            next_coords = node_coords[torch.arange(batch_size), next_node_idx]
            distance = torch.norm(next_coords - current_coords, dim=-1)
            
            predicted_time = self.time_predictor(current_node_emb, next_node_emb, current_hour, distance)
            predicted_times.append(predicted_time)
            
            elapsed_hours = (predicted_time / 3600.0).long()
            current_hour = (current_hour + elapsed_hours) % 24
            
            # 🔥 마스크 업데이트 (방문한 노드 표시)
            mask[torch.arange(batch_size), next_node_idx] = 1
            
            current_node_idx = next_node_idx
            
            # 🔥 핵심: 다음 LSTM 입력은 선택된 노드의 임베딩
            # Teacher Forcing 시에는 정답 노드의 임베딩이 사용됨
            current_input = next_node_emb
        
        routes = torch.stack(routes, dim=1)
        logits_sequence = torch.stack(logits_sequence, dim=1)
        predicted_times = torch.stack(predicted_times, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)
        
        return routes, logits_sequence, predicted_times, attention_weights