import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###############################################################################
# 🔥 완전히 새로 작성된 Pointer Network (Bahdanau Attention + Glimpse)
###############################################################################

class Attention(nn.Module):
    """
    🔥🔥🔥 Dot-Product Attention + Tanh Boosting
    - Scaled Dot-Product: (Q @ K^T) / sqrt(d)
    - Tanh Clipping & 10x Boosting으로 차이 극대화
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Query, Key 변환 (bias 없음!)
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 🔥 가중치를 크게 초기화 (차이 증폭)
        nn.init.xavier_uniform_(self.W_query.weight, gain=2.0)
        nn.init.xavier_uniform_(self.W_key.weight, gain=2.0)
        
    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (batch, hidden_dim) - decoder hidden state
            keys: (batch, num_nodes, hidden_dim) - encoder outputs
            mask: (batch, num_nodes) - True면 해당 노드 선택 불가
            
        Returns:
            logits: (batch, num_nodes) - attention scores
            probs: (batch, num_nodes) - attention weights
        """
        batch_size, num_nodes, _ = keys.shape
        
        # 🔥 [1] Query, Key 변환
        q = self.W_query(query)  # (batch, hidden_dim)
        k = self.W_key(keys)      # (batch, num_nodes, hidden_dim)
        
        # 🔥 [2] Dot-Product Attention Score
        # score = Q @ K^T = (batch, hidden) @ (batch, hidden, nodes) = (batch, nodes)
        q = q.unsqueeze(1)  # (batch, 1, hidden)
        score = torch.bmm(q, k.transpose(1, 2))  # (batch, 1, num_nodes)
        score = score.squeeze(1)  # (batch, num_nodes)
        
        # 🔥 [3] Scaling: sqrt(hidden_dim)으로 나누기 (먼저!)
        score = score / math.sqrt(self.hidden_dim)
        
        # 🔥🔥🔥 [4] 핵심: Tanh Clipping & 10x Boosting!
        # tanh는 값을 -1~1로 정규화, *10은 차이를 극대화
        # 이렇게 해야 Softmax가 노드를 구분할 수 있음!
        logits = 10.0 * torch.tanh(score)
        
        # 🔥 [5] Masking: 방문한 노드는 -inf로
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), -1e9)
        
        # Softmax
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs


class Glimpse(nn.Module):
    """
    Glimpse Layer: Attention으로 Context Vector 생성
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = Attention(hidden_dim)
        
    def forward(self, query, keys, mask=None):
        """
        Returns:
            context: (batch, hidden_dim) - weighted sum of keys
            probs: (batch, num_nodes) - attention weights
        """
        logits, probs = self.attention(query, keys, mask)
        
        # Context = weighted sum of keys
        # probs: (batch, num_nodes) -> (batch, 1, num_nodes)
        # keys: (batch, num_nodes, hidden) 
        context = torch.bmm(probs.unsqueeze(1), keys).squeeze(1)  # (batch, hidden)
        
        return context, probs


class PointerDecoder(nn.Module):
    """
    🔥 완전히 새로 작성된 Pointer Network Decoder
    
    핵심 로직:
    1. LSTM으로 decoder hidden state 생성
    2. Glimpse로 context vector 생성 (encoder outputs 참조)
    3. Pointer Attention으로 다음 노드 선택
    4. 선택된 노드의 임베딩을 다음 스텝 입력으로 사용
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # LSTM Decoder
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Glimpse Layer (context 생성용)
        self.glimpse = Glimpse(hidden_dim)
        
        # Pointer Attention (노드 선택용)
        self.pointer = Attention(hidden_dim)
        
        # Context와 Hidden을 결합
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Time Predictor
        self.time_predictor = TimePredictor(hidden_dim, dropout)
        
        # 초기화
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)
                
    def forward(self, encoder_output, node_coords, start_hour,
                num_nodes_list=None, teacher_route=None, 
                teacher_forcing_ratio=0.5, training=True, debug=False):
        """
        Args:
            encoder_output: (batch, num_nodes, hidden_dim) - GNN/Transformer 출력
            node_coords: (batch, num_nodes, 2) - 노드 좌표
            start_hour: (batch,) - 시작 시간
            num_nodes_list: list of int - 각 샘플의 실제 노드 수
            teacher_route: (batch, num_nodes) - 정답 경로
            teacher_forcing_ratio: float - teacher forcing 비율
            training: bool - 학습 모드 여부
            debug: bool - 디버깅 출력 여부
        """
        batch_size, max_nodes, hidden_dim = encoder_output.shape
        device = encoder_output.device
        
        # 결과 저장용
        routes = []
        all_logits = []
        all_times = []
        all_attn = []
        
        # 🔥 [초기화] LSTM hidden/cell state
        # Encoder output의 평균으로 초기화 (정보 전달)
        h = encoder_output.mean(dim=1)  # (batch, hidden)
        c = torch.zeros_like(h)
        
        # 🔥 [초기화] 첫 입력 = Depot(0번 노드)의 임베딩
        current_input = encoder_output[:, 0, :]  # (batch, hidden)
        current_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        routes.append(current_idx)
        
        # 🔥 [초기화] Mask - Depot(0)과 패딩 노드 마스킹
        mask = torch.zeros(batch_size, max_nodes, device=device)
        mask[:, 0] = 1  # Depot 마스킹
        
        if num_nodes_list is not None:
            for b in range(batch_size):
                n = num_nodes_list[b]
                if n < max_nodes:
                    mask[b, n:] = 1  # 패딩 노드 마스킹
        
        # 최대 스텝 수
        max_steps = max_nodes - 1
        if num_nodes_list is not None:
            max_steps = max(num_nodes_list) - 1
        
        current_hour = start_hour.clone()
        
        # 🔥 [디코딩 루프]
        for step in range(max_steps):
            # [Step 1] LSTM으로 hidden state 업데이트
            h, c = self.lstm(current_input, (h, c))
            
            # [Step 2] Glimpse로 context 생성 (encoder outputs 참조)
            context, glimpse_attn = self.glimpse(h, encoder_output, mask)
            
            # [Step 3] Hidden과 Context 결합
            query = self.hidden_out(torch.cat([h, context], dim=-1))
            
            # [Step 4] Pointer Attention으로 노드 선택
            logits, probs = self.pointer(query, encoder_output, mask)
            
            # Loss 계산용 (마스킹 전 logits)
            raw_logits, _ = self.pointer(query, encoder_output, mask=None)
            all_logits.append(raw_logits)
            all_attn.append(probs)
            
            # 🔥 디버깅: 첫 스텝의 Attention Score 출력
            if debug and step == 0:
                print("\n" + "="*60)
                print("🔍 [DEBUG] Pointer Attention Score (Step 0)")
                print("="*60)
                n = num_nodes_list[0] if num_nodes_list else max_nodes
                print(f"Raw logits (마스킹 전) 샘플0, 노드 0~{min(5,n)-1}:")
                print(f"   {raw_logits[0, :min(5,n)].detach().cpu().numpy()}")
                print(f"Logits min/max: {raw_logits[0,:n].min().item():.4f} / {raw_logits[0,:n].max().item():.4f}")
                print(f"Probs (softmax 후) 샘플0, 노드 0~{min(5,n)-1}:")
                print(f"   {probs[0, :min(5,n)].detach().cpu().numpy()}")
                print(f"Probs max: {probs[0,:n].max().item():.4f} (uniform이면 ~{1.0/n:.4f})")
                
                # Uniform 체크
                uniform_prob = 1.0 / n
                max_prob = probs[0,:n].max().item()
                if max_prob < uniform_prob * 1.5:
                    print("⚠️ 경고: Softmax가 거의 Uniform! Attention 차이가 부족함")
                else:
                    print("✅ Softmax 분포가 뾰족함! Attention 작동 중")
                print("="*60 + "\n")
            
            # 🔥 [Step 5] 다음 노드 선택
            if training and teacher_route is not None:
                # Teacher Forcing
                if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                    next_idx = teacher_route[:, step + 1]
                else:
                    # Sampling
                    next_idx = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # Greedy (Inference)
                next_idx = probs.argmax(dim=-1)
            
            routes.append(next_idx)
            
            # 🔥 [Step 6] 시간 예측
            current_emb = encoder_output[torch.arange(batch_size), current_idx]
            next_emb = encoder_output[torch.arange(batch_size), next_idx]
            
            curr_coord = node_coords[torch.arange(batch_size), current_idx]
            next_coord = node_coords[torch.arange(batch_size), next_idx]
            distance = torch.norm(next_coord - curr_coord, dim=-1)
            
            pred_time = self.time_predictor(current_emb, next_emb, current_hour, distance)
            all_times.append(pred_time)
            
            # 시간 업데이트
            elapsed = (pred_time / 3600.0).long()
            current_hour = (current_hour + elapsed) % 24
            
            # 🔥 [Step 7] 다음 스텝 준비
            mask[torch.arange(batch_size), next_idx] = 1  # 방문 마스킹
            current_idx = next_idx
            
            # 🔥 [Input Feeding] 선택된 노드의 임베딩을 다음 입력으로!
            current_input = encoder_output[torch.arange(batch_size), next_idx]
        
        # 결과 스택
        routes = torch.stack(routes, dim=1)  # (batch, num_nodes)
        all_logits = torch.stack(all_logits, dim=1)  # (batch, steps, num_nodes)
        all_times = torch.stack(all_times, dim=1)  # (batch, steps)
        all_attn = torch.stack(all_attn, dim=1)  # (batch, steps, num_nodes)
        
        return routes, all_logits, all_times, all_attn


class TimePredictor(nn.Module):
    """시간 예측기 (단순화)"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.hour_embed = nn.Embedding(24, hidden_dim // 4)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 출력 스케일
        self.scale = nn.Parameter(torch.tensor(300.0))
        self.bias = nn.Parameter(torch.tensor(200.0))
        
    def forward(self, curr_emb, next_emb, hour, distance=None):
        batch_size = curr_emb.size(0)
        
        hour_emb = self.hour_embed(hour)
        
        if distance is None:
            dist = torch.zeros(batch_size, 1, device=curr_emb.device)
        else:
            dist = distance.unsqueeze(-1)
        
        x = torch.cat([curr_emb, next_emb, hour_emb, dist], dim=-1)
        raw = self.mlp(x).squeeze(-1)
        
        # Softplus + Scale
        time = F.softplus(raw) * torch.abs(self.scale) + torch.abs(self.bias)
        time = time + dist.squeeze(-1) * 500.0  # 거리 보정
        
        return torch.clamp(time, min=60.0, max=7200.0)