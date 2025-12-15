import torch
import torch.nn as nn
import torch.nn.functional as F
import math


###############################################################################
# 🔬 CT Scan 모드: 텐서 통계 출력 헬퍼 함수
###############################################################################

def log_tensor_stats(name, tensor, indent=0):
    """
    텐서의 통계치를 출력하는 헬퍼 함수
    NaN/Inf 감지 시 경고 출력
    """
    prefix = "   " * indent
    
    # detach하고 float으로 변환
    t = tensor.detach().float()
    
    # NaN/Inf 체크
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    
    if has_nan:
        print(f"{prefix}🚨 [{name}] NaN Detected!")
        return
    if has_inf:
        print(f"{prefix}🚨 [{name}] Inf Detected!")
        return
    
    # 통계 계산
    t_min = t.min().item()
    t_max = t.max().item()
    t_mean = t.mean().item()
    t_std = t.std().item()
    
    # 경고 조건
    warning = ""
    if abs(t_max) > 100 or abs(t_min) > 100:
        warning = " ⚠️ 값이 큼!"
    elif t_std < 1e-6:
        warning = " ⚠️ std가 0에 가까움!"
    
    print(f"{prefix}📊 [{name}] min:{t_min:+.4f}, max:{t_max:+.4f}, mean:{t_mean:+.4f}, std:{t_std:.4f}{warning}")


###############################################################################
# 🔥 완전히 새로 작성된 Pointer Network (Bahdanau Attention + Glimpse)
###############################################################################

class Attention(nn.Module):
    """
    🔥 Pure Scaled Dot-Product Attention (가장 단순한 형태)
    - tanh, clipping, boosting 전부 제거
    - 오직 score = (Q @ K^T) / sqrt(d) 만 사용
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Query, Key 변환
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 표준 Xavier 초기화
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_key.weight)
        
        # 디버깅용 플래그
        self.debug_printed = False
        
    def forward(self, query, keys, mask=None, debug=False):
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
        
        # ============================================================
        # 🔬 CT Scan Point (3): Query, Key 생성 직후
        # ============================================================
        q = self.W_query(query)  # (batch, hidden_dim)
        k = self.W_key(keys)      # (batch, num_nodes, hidden_dim)
        
        if debug:
            print("\n" + "="*60)
            print("🔬 CT SCAN: Attention 내부")
            print("="*60)
            log_tensor_stats("Query (W_q @ h)", q, indent=1)
            log_tensor_stats("Key (W_k @ enc)", k, indent=1)
        
        # ============================================================
        # 🔬 CT Scan Point (4): Attention Score (Scaling 전/후)
        # ============================================================
        q = q.unsqueeze(1)  # (batch, 1, hidden)
        score_raw = torch.bmm(q, k.transpose(1, 2))  # (batch, 1, num_nodes)
        score_raw = score_raw.squeeze(1)  # (batch, num_nodes)
        
        if debug:
            log_tensor_stats("Score (Q@K^T) - Scaling 전", score_raw, indent=1)
        
        # Scaling: 1 / sqrt(d)
        score_scaled = score_raw / math.sqrt(self.hidden_dim)
        
        if debug:
            log_tensor_stats("Score (Q@K^T/√d) - Scaling 후", score_scaled, indent=1)
        
        # 값 범위 제한 (Overflow 방지)
        logits = torch.clamp(score_scaled, min=-50.0, max=50.0)
        
        if debug:
            log_tensor_stats("Logits (clamp 후)", logits, indent=1)
        
        # Masking
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), float('-inf'))
        
        # ============================================================
        # 🔬 CT Scan Point (5): Softmax 직후
        # ============================================================
        # 안전한 Softmax (max를 빼서 overflow 방지)
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        probs = F.softmax(logits_stable, dim=-1)
        
        if debug:
            log_tensor_stats("Probs (softmax 후)", probs, indent=1)
            
            # 분포 분석
            max_prob = probs[0].max().item()
            uniform_prob = 1.0 / num_nodes
            print(f"   📈 Probs max: {max_prob:.4f} (uniform={uniform_prob:.4f})")
            
            if max_prob < uniform_prob * 1.5:
                print("   ❌ Softmax가 Uniform! → Attention 실패")
            elif max_prob > 0.5:
                print("   ✅ Softmax가 뾰족함! → Attention 성공")
            else:
                print("   ⚠️ Softmax가 약간 뾰족함")
            print("="*60)
        
        return logits, probs


class Glimpse(nn.Module):
    """
    Glimpse Layer: Attention으로 Context Vector 생성
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = Attention(hidden_dim)
        
    def forward(self, query, keys, mask=None, debug=False):
        """
        Returns:
            context: (batch, hidden_dim) - weighted sum of keys
            probs: (batch, num_nodes) - attention weights
        """
        logits, probs = self.attention(query, keys, mask, debug=debug)
        
        # Context = weighted sum of keys
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
            
            # 🔬 CT Scan: LSTM 출력 (첫 스텝만)
            if debug and step == 0:
                print("\n" + "="*60)
                print("🔬 CT SCAN: Decoder 내부 (Step 0)")
                print("="*60)
                log_tensor_stats("LSTM Input", current_input, indent=1)
                log_tensor_stats("LSTM Hidden (h)", h, indent=1)
            
            # [Step 2] Glimpse로 context 생성
            context, glimpse_attn = self.glimpse(h, encoder_output, mask, debug=(debug and step==0))
            
            if debug and step == 0:
                log_tensor_stats("Glimpse Context", context, indent=1)
            
            # [Step 3] Hidden과 Context 결합
            query = self.hidden_out(torch.cat([h, context], dim=-1))
            
            if debug and step == 0:
                log_tensor_stats("Query (h + context)", query, indent=1)
                print("="*60)
            
            # [Step 4] Pointer Attention으로 노드 선택
            logits, probs = self.pointer(query, encoder_output, mask, debug=(debug and step==0))
            
            # Loss 계산용 (마스킹 전 logits)
            raw_logits, _ = self.pointer(query, encoder_output, mask=None, debug=False)
            all_logits.append(raw_logits)
            all_attn.append(probs)
            
            # 🔥 디버깅: 첫 스텝의 결과 요약
            if debug and step == 0:
                n = num_nodes_list[0] if num_nodes_list else max_nodes
                print(f"\n📊 Step 0 요약:")
                print(f"   Probs 샘플0: {probs[0, :min(5,n)].detach().cpu().numpy()}")
                print(f"   Probs max: {probs[0,:n].max().item():.4f} (uniform={1.0/n:.4f})")
                
                # 선택될 노드 예측
                pred_node = probs[0].argmax().item()
                print(f"   예측 노드: {pred_node}")
            
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