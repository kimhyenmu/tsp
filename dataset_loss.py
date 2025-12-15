import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from model.topmodel import HybridRoutingModel
# ==========================================

# 1. Dataset 클래스
# ==========================================
class DeliveryDataset(Dataset):
    """
    배송 경로 데이터셋
    """
    def __init__(self, data_path):
        print(f"📂 데이터 로드 중: {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"   ✅ {len(self.data):,}개 샘플 로드 완료")
        
        # 통계 출력
        num_nodes = [d['num_nodes'] for d in self.data]
        print(f"   - 평균 노드 수: {np.mean(num_nodes):.1f}")
        print(f"   - 노드 범위: {min(num_nodes)} ~ {max(num_nodes)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        return {
            # 입력
            'node_features': torch.FloatTensor(sample['node_features']),
            'start_hour': torch.LongTensor([sample['start_hour']]).squeeze(),
            'start_minute': torch.LongTensor([sample['start_minute']]).squeeze(),
            'day_of_week': torch.LongTensor([sample['day_of_week']]).squeeze(),
            'traffic_profile': torch.FloatTensor(sample['traffic_profile']),
            'num_nodes': sample['num_nodes'],
            
            # 정답 라벨
            'actual_route': torch.LongTensor(sample['actual_route']),
            'actual_times': torch.FloatTensor(sample['actual_times']),
            'actual_total_time': sample['actual_total_time'],
            
            # Baseline (비교용)
            'baseline_route': torch.LongTensor(sample['baseline_route']),
            'baseline_total_time': sample['baseline_total_time'],
            
            # 메타
            'trip_id': sample['trip_id']
        }


# ==========================================
# 2. Collate 함수 (배치 처리)
# ==========================================
def collate_fn(batch):
    """
    가변 길이 시퀀스를 배치로 묶기
    """
    # 최대 노드 수 찾기
    max_nodes = max(sample['num_nodes'] for sample in batch)
    batch_size = len(batch)
    
    # 텐서 초기화
    node_features = torch.zeros(batch_size, max_nodes, 2)
    actual_routes = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    actual_times = torch.zeros(batch_size, max_nodes - 1)
    
    start_hours = []
    start_minutes = []
    day_of_weeks = []
    traffic_profiles = []
    num_nodes_list = []
    
    baseline_routes = []
    actual_total_times = []
    baseline_total_times = []
    trip_ids = []
    
    for i, sample in enumerate(batch):
        n = sample['num_nodes']
        
        node_features[i, :n, :] = sample['node_features']
        actual_routes[i, :n] = sample['actual_route']
        actual_times[i, :n-1] = sample['actual_times']
        
        start_hours.append(sample['start_hour'])
        start_minutes.append(sample['start_minute'])
        day_of_weeks.append(sample['day_of_week'])
        traffic_profiles.append(sample['traffic_profile'])
        
        num_nodes_list.append(n)
        baseline_routes.append(sample['baseline_route'])
        actual_total_times.append(sample['actual_total_time'])
        baseline_total_times.append(sample['baseline_total_time'])
        trip_ids.append(sample['trip_id'])
    
    return {
        'node_features': node_features,
        'start_hour': torch.stack(start_hours),
        'start_minute': torch.stack(start_minutes),
        'day_of_week': torch.stack(day_of_weeks),
        'traffic_profile': torch.stack(traffic_profiles),
        'actual_route': actual_routes,
        'actual_times': actual_times,
        'num_nodes': num_nodes_list,
        'baseline_routes': baseline_routes,
        'actual_total_times': actual_total_times,
        'baseline_total_times': baseline_total_times,
        'trip_ids': trip_ids
    }


# ==========================================
# 3. Loss 함수
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, route_weight=1.0, time_weight=0.1, smoothing=0.1):
        super().__init__()
        self.route_weight = route_weight
        self.time_weight = time_weight
        self.smoothing = smoothing

    def forward(self, predictions, targets, num_nodes_list):
        logits = predictions['logits']
        predicted_times = predictions['predicted_times']
        actual_route = targets['actual_route']
        actual_times = targets['actual_times']
        
        batch_size, max_steps, max_nodes = logits.shape
        route_loss = torch.tensor(0.0, device=logits.device)
        correct_predictions = 0
        total_predictions = 0
        
        for b in range(batch_size):
            n = num_nodes_list[b]
            
            # 🔥 핵심 수정: 각 step에서 방문한 노드를 추적
            visited_mask = torch.zeros(n, device=logits.device)
            visited_mask[actual_route[b, 0]] = 1  # depot 마스킹
            
            for step in range(n - 1):
                target = actual_route[b, step + 1]
                
                # 유효 노드만 사용 (패딩 제외)
                step_logits = logits[b, step, :n].clone()
                
                # 🔥 이미 방문한 노드의 logits는 -inf로 설정되어 있음
                # CrossEntropy가 올바르게 계산되도록 함
                # (마스킹된 노드는 softmax 후 거의 0이 됨)
                
                step_loss = F.cross_entropy(
                    step_logits.unsqueeze(0), 
                    target.unsqueeze(0),
                    label_smoothing=self.smoothing  # 이제 label smoothing 사용 가능
                )
                
                route_loss += step_loss
                
                # 🔥 argmax 시 마스킹된 값(-inf)은 자동으로 제외됨
                pred_node = step_logits.argmax()
                if pred_node == target:
                    correct_predictions += 1
                total_predictions += 1
                
                # 다음 step을 위해 방문 마스크 업데이트
                visited_mask[target] = 1
        
        if total_predictions > 0:
            route_loss = route_loss / total_predictions
            route_accuracy = correct_predictions / total_predictions
        else:
            route_accuracy = 0

        # 🔥 개선된 Time Loss
        # - 로그 스케일 적용으로 큰 값과 작은 값의 균형 맞춤
        # - Huber Loss로 이상치에 강건하게
        time_loss_val = torch.tensor(0.0, device=logits.device)
        time_mae_val = torch.tensor(0.0, device=logits.device)
        total_time_predictions = 0

        for b in range(batch_size):
            n = num_nodes_list[b]
            pred = predicted_times[b, :n-1]
            target = actual_times[b, :n-1]
            
            # 음수 방지
            pred = torch.clamp(pred, min=1.0)
            target = torch.clamp(target, min=1.0)
            
            # 🔥 로그 스케일 변환 (큰 값의 영향력 감소)
            # log(시간)으로 변환하면 100초와 1000초의 차이가 선형적으로 됨
            pred_log = torch.log(pred + 1.0)
            target_log = torch.log(target + 1.0)
            
            # Huber Loss (Smooth L1) 적용
            loss_val = F.smooth_l1_loss(pred_log, target_log)
            
            time_loss_val = time_loss_val + loss_val
            time_mae_val = time_mae_val + F.l1_loss(pred, target)
            total_time_predictions += 1

        if total_time_predictions > 0:
            time_loss_val = time_loss_val / total_time_predictions
            time_mae_val = time_mae_val / total_time_predictions

        # 🔥 Loss 스케일 조정
        # Route Loss와 Time Loss의 스케일을 맞춤
        total_loss = self.route_weight * route_loss + self.time_weight * time_loss_val

        return {
            'total_loss': total_loss,
            'route_loss': route_loss,
            'time_loss': time_loss_val,
            'time_mae': time_mae_val,
            'route_accuracy': route_accuracy
        }


# ==========================================
# 4. 평가 메트릭
# ==========================================
class EvaluationMetrics:
    """모델 성능 평가 지표"""
    
    @staticmethod
    def kendall_tau(pred_route, actual_route):
        """
        Kendall's Tau 상관계수
        경로 순서의 유사도 측정 (-1 ~ 1)
        """
        pred_route = pred_route.cpu().numpy()
        actual_route = actual_route.cpu().numpy()
        
        n = len(pred_route)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # 실제 경로에서의 위치 찾기
                actual_i_pos = np.where(actual_route == pred_route[i])[0]
                actual_j_pos = np.where(actual_route == pred_route[j])[0]
                
                if len(actual_i_pos) == 0 or len(actual_j_pos) == 0:
                    continue
                
                actual_i_pos = actual_i_pos[0]
                actual_j_pos = actual_j_pos[0]
                
                if actual_i_pos < actual_j_pos:
                    concordant += 1
                else:
                    discordant += 1
        
        if concordant + discordant == 0:
            return 0
        
        tau = (concordant - discordant) / (n * (n - 1) / 2)
        return tau
    
    @staticmethod
    def accuracy_at_k(pred_route, actual_route, k=5):
        """처음 k개 노드의 일치율"""
        pred_route = pred_route.cpu().numpy()
        actual_route = actual_route.cpu().numpy()
        
        k = min(k, len(pred_route))
        matches = sum(pred_route[i] == actual_route[i] for i in range(k))
        return matches / k
    
    @staticmethod
    def time_metrics(pred_times, actual_times):
        """
        시간 예측 메트릭
        Returns: MAE, RMSE, MAPE
        """
        pred_times = pred_times.cpu()
        actual_times = actual_times.cpu()
        
        # MAE
        mae = (pred_times - actual_times).abs().mean().item()
        
        # RMSE
        rmse = ((pred_times - actual_times) ** 2).mean().sqrt().item()
        
        # MAPE (%)
        mape = ((pred_times - actual_times).abs() / (actual_times + 1e-6)).mean().item() * 100
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    @staticmethod
    def compute_all_metrics(predictions, targets, num_nodes_list):
        """모든 메트릭 계산"""
        batch_size = len(num_nodes_list)
        
        metrics = {
            'kendall_tau': [],
            'acc_at_5': [],
            'time_mae': [],
            'time_rmse': [],
            'time_mape': [],
            'total_time_error': []  # 총 시간 오차 (%)
        }
        
        for b in range(batch_size):
            n = num_nodes_list[b]
            
            pred_route = predictions['routes'][b, :n]
            actual_route = targets['actual_route'][b, :n]
            
            pred_times = predictions['predicted_times'][b, :n-1]
            actual_times = targets['actual_times'][b, :n-1]
            
            # Route metrics
            tau = EvaluationMetrics.kendall_tau(pred_route, actual_route)
            acc_k = EvaluationMetrics.accuracy_at_k(pred_route, actual_route, k=5)
            
            # Time metrics
            time_m = EvaluationMetrics.time_metrics(pred_times, actual_times)
            
            # Total time error
            pred_total = pred_times.sum().item()
            actual_total = actual_times.sum().item()
            total_error = abs(pred_total - actual_total) / actual_total * 100
            
            metrics['kendall_tau'].append(tau)
            metrics['acc_at_5'].append(acc_k)
            metrics['time_mae'].append(time_m['MAE'])
            metrics['time_rmse'].append(time_m['RMSE'])
            metrics['time_mape'].append(time_m['MAPE'])
            metrics['total_time_error'].append(total_error)
        
        # 평균 계산
        return {k: np.mean(v) for k, v in metrics.items()}


# ==========================================
# 5. 테스트
# ==========================================
