"""
🔬 과적합 테스트 (Overfitting Sanity Check)
============================================
소규모 데이터(10개)에 대해 Loss가 0에 수렴하는지 확인
- Loss가 0에 수렴하면: 모델 구조 OK, 데이터/하이퍼파라미터 문제
- Loss가 수렴 안 하면: 모델 구조 또는 Gradient Flow 문제

사용법:
    python overfit_test.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.topmodel import HybridRoutingModel
from dataset_loss import DeliveryDataset, collate_fn


def create_mini_dataset(data_path, num_samples=10):
    """
    전체 데이터에서 num_samples개만 추출
    """
    print(f"📂 데이터 로드: {data_path}")
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)
    
    # 노드 수가 적당한 샘플 선택 (10~20개 노드)
    filtered = [d for d in full_data if 10 <= d['num_nodes'] <= 20]
    
    if len(filtered) < num_samples:
        filtered = full_data[:num_samples]
    else:
        filtered = filtered[:num_samples]
    
    print(f"✅ {len(filtered)}개 샘플 선택")
    for i, d in enumerate(filtered):
        print(f"   샘플 {i}: 노드 {d['num_nodes']}개, 경로 {d['actual_route'][:5]}...")
    
    return filtered


class MiniDataset(torch.utils.data.Dataset):
    """미니 데이터셋"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'node_features': torch.FloatTensor(sample['node_features']),
            'start_hour': torch.LongTensor([sample['start_hour']]).squeeze(),
            'start_minute': torch.LongTensor([sample['start_minute']]).squeeze(),
            'day_of_week': torch.LongTensor([sample['day_of_week']]).squeeze(),
            'traffic_profile': torch.FloatTensor(sample['traffic_profile']),
            'num_nodes': sample['num_nodes'],
            'actual_route': torch.LongTensor(sample['actual_route']),
            'actual_times': torch.FloatTensor(sample['actual_times']),
            'actual_total_time': sample['actual_total_time'],
            'baseline_route': torch.LongTensor(sample['baseline_route']),
            'baseline_total_time': sample['baseline_total_time'],
            'trip_id': sample.get('trip_id', idx)
        }


def compute_loss_detailed(predictions, targets, num_nodes_list, device):
    """
    상세 Loss 계산 (디버깅용)
    """
    logits = predictions['logits']
    predicted_times = predictions['predicted_times']
    actual_route = targets['actual_route']
    actual_times = targets['actual_times']
    
    batch_size = logits.size(0)
    
    # Route Loss
    route_losses = []
    correct = 0
    total = 0
    
    for b in range(batch_size):
        n = num_nodes_list[b]
        
        for step in range(n - 1):
            target = actual_route[b, step + 1]
            step_logits = logits[b, step, :n]
            
            # CrossEntropy
            loss = F.cross_entropy(step_logits.unsqueeze(0), target.unsqueeze(0))
            route_losses.append(loss)
            
            # Accuracy
            if step_logits.argmax() == target:
                correct += 1
            total += 1
    
    route_loss = torch.stack(route_losses).mean()
    route_acc = correct / total if total > 0 else 0
    
    # Time Loss
    time_losses = []
    for b in range(batch_size):
        n = num_nodes_list[b]
        pred = predicted_times[b, :n-1]
        target = actual_times[b, :n-1]
        
        pred = torch.clamp(pred, min=1.0)
        target = torch.clamp(target, min=1.0)
        
        loss = F.smooth_l1_loss(torch.log1p(pred), torch.log1p(target))
        time_losses.append(loss)
    
    time_loss = torch.stack(time_losses).mean() if time_losses else torch.tensor(0.0, device=device)
    
    # Total
    total_loss = route_loss + 0.05 * time_loss
    
    return {
        'total_loss': total_loss,
        'route_loss': route_loss,
        'time_loss': time_loss,
        'route_acc': route_acc
    }


def check_gradients(model):
    """
    Gradient 흐름 확인
    """
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name] = {
                'norm': grad_norm,
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
    return grad_info


def overfit_test(
    data_path='./model_data/train_data.pkl',
    num_samples=10,
    num_epochs=500,
    learning_rate=1e-3,
    print_every=10
):
    """
    과적합 테스트 메인 함수
    """
    print("\n" + "="*70)
    print("🔬 과적합 테스트 (Overfitting Sanity Check)")
    print("="*70)
    print(f"샘플 수: {num_samples}")
    print(f"에폭 수: {num_epochs}")
    print(f"학습률: {learning_rate}")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 미니 데이터셋 생성
    mini_data = create_mini_dataset(data_path, num_samples)
    dataset = MiniDataset(mini_data)
    
    # 전체 데이터를 하나의 배치로 (과적합 테스트)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_samples,  # 전체를 하나의 배치로
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 배치 하나만 가져오기
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    print(f"\n📊 배치 정보:")
    print(f"   - node_features shape: {batch['node_features'].shape}")
    print(f"   - actual_route shape: {batch['actual_route'].shape}")
    print(f"   - num_nodes: {batch['num_nodes']}")
    
    # 모델 초기화
    print("\n🧠 모델 초기화...")
    model = HybridRoutingModel(
        node_dim=2,
        hidden_dim=128,
        gnn_layers=3,
        gnn_heads=4,
        tf_layers=4,
        tf_heads=8,
        tf_ff_dim=512,
        pointer_heads=4,
        dropout=0.0,  # 🔥 과적합 테스트에서는 dropout=0
        k_neighbors=10
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - 파라미터 수: {total_params:,}")
    
    # Optimizer (과적합 테스트에는 높은 학습률)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 기록
    history = {
        'total_loss': [],
        'route_loss': [],
        'time_loss': [],
        'route_acc': []
    }
    
    # 학습 루프
    print("\n🚀 과적합 테스트 시작...")
    print("-" * 70)
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward (항상 teacher_forcing=1.0)
        predictions = model(batch, training=True, teacher_forcing_ratio=1.0)
        
        # Loss 계산
        targets = {
            'actual_route': batch['actual_route'],
            'actual_times': batch['actual_times']
        }
        
        loss_dict = compute_loss_detailed(predictions, targets, batch['num_nodes'], device)
        
        # Backward
        loss_dict['total_loss'].backward()
        
        # Gradient 확인 (첫 에폭과 중간중간)
        if epoch == 0 or (epoch + 1) % 100 == 0:
            grad_info = check_gradients(model)
            nan_grads = [k for k, v in grad_info.items() if v['has_nan']]
            inf_grads = [k for k, v in grad_info.items() if v['has_inf']]
            zero_grads = [k for k, v in grad_info.items() if v['norm'] < 1e-7]
            
            if nan_grads:
                print(f"⚠️ NaN Gradients: {nan_grads}")
            if inf_grads:
                print(f"⚠️ Inf Gradients: {inf_grads}")
            if zero_grads and epoch == 0:
                print(f"⚠️ Zero Gradients (가능한 gradient 끊김): {zero_grads[:5]}...")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 기록
        history['total_loss'].append(loss_dict['total_loss'].item())
        history['route_loss'].append(loss_dict['route_loss'].item())
        history['time_loss'].append(loss_dict['time_loss'].item())
        history['route_acc'].append(loss_dict['route_acc'])
        
        # 출력
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Loss: {loss_dict['total_loss'].item():.6f} | "
                  f"Route Loss: {loss_dict['route_loss'].item():.6f} | "
                  f"Route Acc: {loss_dict['route_acc']:.3f} | "
                  f"Time Loss: {loss_dict['time_loss'].item():.6f}")
        
        # 조기 종료 조건
        if loss_dict['total_loss'].item() < 0.01 and loss_dict['route_acc'] > 0.99:
            print(f"\n🎉 Loss가 0.01 미만, Accuracy가 99% 이상 도달!")
            print(f"   → 모델 구조에 문제 없음!")
            break
    
    # 최종 결과
    print("\n" + "="*70)
    print("📊 최종 결과")
    print("="*70)
    print(f"최종 Loss: {history['total_loss'][-1]:.6f}")
    print(f"최종 Route Loss: {history['route_loss'][-1]:.6f}")
    print(f"최종 Route Accuracy: {history['route_acc'][-1]:.3f}")
    print(f"최종 Time Loss: {history['time_loss'][-1]:.6f}")
    
    # 판정
    print("\n" + "="*70)
    print("🔍 진단 결과")
    print("="*70)
    
    final_loss = history['total_loss'][-1]
    final_acc = history['route_acc'][-1]
    
    if final_loss < 0.1 and final_acc > 0.95:
        print("✅ 과적합 성공! 모델 구조는 정상입니다.")
        print("   → 문제는 데이터 양, 하이퍼파라미터, 또는 정규화에 있습니다.")
    elif final_loss < 0.5 and final_acc > 0.7:
        print("⚠️ 부분적 과적합. 모델이 학습은 하지만 완전히 수렴하지 못함.")
        print("   → 학습률 증가 또는 에폭 수 증가 시도")
    else:
        print("❌ 과적합 실패! 모델 구조 또는 Gradient Flow에 문제가 있습니다.")
        print("   → 아래 항목들을 확인하세요:")
        print("   1. Loss 함수가 올바르게 계산되는지")
        print("   2. Gradient가 모든 레이어로 흐르는지")
        print("   3. 모듈 간 연결이 끊어지지 않았는지")
    
    # Loss 곡선 그래프
    print("\n📈 Loss 곡선 저장 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['route_loss'])
    axes[0, 1].set_title('Route Loss (CrossEntropy)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['route_acc'])
    axes[1, 0].set_title('Route Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['time_loss'])
    axes[1, 1].set_title('Time Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('overfit_test_result.png', dpi=150)
    print(f"   ✅ 저장 완료: overfit_test_result.png")
    
    # 예측 결과 상세 출력
    print("\n" + "="*70)
    print("🔎 예측 결과 상세 (마지막 에폭)")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        predictions = model(batch, training=False)
    
    for i in range(min(3, num_samples)):
        n = batch['num_nodes'][i]
        actual = batch['actual_route'][i, :n].cpu().numpy()
        predicted = predictions['routes'][i, :n].cpu().numpy()
        
        match = (actual == predicted).sum()
        
        print(f"\n샘플 {i} (노드 {n}개):")
        print(f"   정답 경로:    {actual}")
        print(f"   예측 경로:    {predicted}")
        print(f"   일치율:       {match}/{n} ({match/n*100:.1f}%)")
    
    return history


if __name__ == "__main__":
    history = overfit_test(
        data_path='./model_data/train_data.pkl',
        num_samples=10,
        num_epochs=500,
        learning_rate=1e-3,
        print_every=20
    )
