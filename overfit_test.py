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


# 🔥 Xavier 초기화 함수 (Gain 증폭!)
def init_weights(module):
    """
    Xavier/Glorot 초기화 적용 (강화 버전)
    - Linear: Xavier Uniform with gain=2.0 (차이를 벌림)
    - Embedding: Normal(0, 0.1) (더 큰 분산)
    - LayerNorm: weight=1, bias=0
    - Bias: 0으로 (차이를 덮지 않게)
    """
    # 🔥 simple_embedding은 이미 별도로 초기화했으므로 건너뜀
    if hasattr(module, '_custom_initialized'):
        return
        
    if isinstance(module, nn.Linear):
        # 🔥 gain=2.0으로 가중치 차이 증폭
        nn.init.xavier_uniform_(module.weight, gain=2.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)  # 🔥 bias=0 (차이를 덮지 않게)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.1)  # 🔥 더 큰 std
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTMCell):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param, gain=2.0)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param, gain=2.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


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
    🔥 Route Loss만 계산 (과적합 테스트 단순화)
    """
    logits = predictions['logits']
    predicted_times = predictions['predicted_times']
    actual_route = targets['actual_route']
    actual_times = targets['actual_times']
    
    batch_size = logits.size(0)
    
    # Route Loss (CrossEntropy)
    route_losses = []
    correct = 0
    total = 0
    
    for b in range(batch_size):
        n = num_nodes_list[b]
        
        for step in range(n - 1):
            target = actual_route[b, step + 1]
            
            # 🔥 유효 노드만 사용 (패딩 제외)
            step_logits = logits[b, step, :n]
            
            # CrossEntropy Loss
            loss = F.cross_entropy(step_logits.unsqueeze(0), target.unsqueeze(0))
            route_losses.append(loss)
            
            # Accuracy
            if step_logits.argmax() == target:
                correct += 1
            total += 1
    
    # 🔥 모든 loss를 합쳐서 평균
    route_loss = torch.stack(route_losses).mean()
    route_acc = correct / total if total > 0 else 0
    
    # Time Loss (간소화)
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
    
    # 🔥 Route Loss에 더 집중 (과적합 테스트에서는 경로 예측이 핵심)
    total_loss = route_loss + 0.01 * time_loss
    
    return {
        'total_loss': total_loss,
        'route_loss': route_loss,
        'time_loss': time_loss,
        'route_acc': route_acc
    }


def check_gradients(model, verbose=False):
    """
    Gradient 흐름 확인 (상세 버전)
    """
    grad_info = {}
    module_grads = {
        'gnn_encoder': [],
        'context_encoder': [],
        'pointer_decoder': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name] = {
                'norm': grad_norm,
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
            
            # 모듈별 분류
            for module_name in module_grads.keys():
                if module_name in name:
                    module_grads[module_name].append(grad_norm)
                    break
        else:
            grad_info[name] = {'norm': 0, 'has_nan': False, 'has_inf': False}
            for module_name in module_grads.keys():
                if module_name in name:
                    module_grads[module_name].append(0)
                    break
    
    if verbose:
        print("\n📊 모듈별 Gradient 평균:")
        for module_name, norms in module_grads.items():
            if norms:
                avg_norm = sum(norms) / len(norms)
                print(f"   {module_name}: {avg_norm:.6f} (params: {len(norms)})")
    
    return grad_info, module_grads


def overfit_test(
    data_path='./model_data/train_data.pkl',
    num_samples=10,
    num_epochs=500,
    learning_rate=0.001,  # 🔥 기본 LR (x10 부스트 적용됨)
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
    print(f"Teacher Forcing: 100% (강제)")
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
    
    # 🔥 정답 경로 출력 (디버깅용)
    print(f"\n📍 정답 경로 샘플:")
    for i in range(min(3, num_samples)):
        n = batch['num_nodes'][i]
        route = batch['actual_route'][i, :n].cpu().numpy()
        print(f"   샘플 {i}: {route}")
    
    # 모델 초기화
    print("\n🧠 모델 초기화...")
    
    # 🔥🔥🔥 GNN Bypass 모드 (True = GNN 끄고 단순 Linear만 사용)
    BYPASS_GNN = True  # ⬅️ True: GNN 비활성화, False: GNN 사용
    
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
        k_neighbors=10,
        bypass_gnn=BYPASS_GNN  # 🔥 GNN Bypass 설정
    ).to(device)
    
    if BYPASS_GNN:
        print("   ⚠️ GNN BYPASS 모드 활성화! (단순 Linear Embedding만 사용)")
    else:
        print("   ✅ GNN 모드 활성화")
    
    # 🔥 Xavier 초기화 강제 적용
    print("   - Xavier 초기화 적용 중...")
    model.apply(init_weights)
    
    # 🔥 simple_embedding은 gain=10으로 다시 강제 초기화 (매우 중요!)
    if BYPASS_GNN:
        print("   - Simple Embedding 가중치 10배 증폭!")
        nn.init.xavier_uniform_(model.simple_embedding.weight, gain=10.0)
        
        # 🔥 초기화 후 가중치 통계 출력
        w = model.simple_embedding.weight
        print(f"   - 가중치 shape: {w.shape}")
        print(f"   - 가중치 mean: {w.mean().item():.4f}, std: {w.std().item():.4f}")
        print(f"   - 가중치 min/max: {w.min().item():.4f} / {w.max().item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - 파라미터 수: {total_params:,}")
    
    # 🔥 Optimizer: Learning Rate 10배 부스팅!
    boosted_lr = learning_rate * 10  # 0.005 -> 0.05
    print(f"   - Learning Rate: {boosted_lr} (10배 부스트)")
    optimizer = optim.Adam(model.parameters(), lr=boosted_lr, weight_decay=0)
    
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
    
    # 🔥 디버깅: Encoder 출력 확인 (핵심!)
    print("\n🔍 Encoder 출력 디버깅...")
    optimizer.zero_grad()
    
    # debug=True로 forward 호출하여 임베딩 값 출력
    test_pred = model(batch, training=True, teacher_forcing_ratio=1.0, debug=True)
    
    test_targets = {'actual_route': batch['actual_route'], 'actual_times': batch['actual_times']}
    test_loss = compute_loss_detailed(test_pred, test_targets, batch['num_nodes'], device)
    test_loss['total_loss'].backward()
    
    grad_info, module_grads = check_gradients(model, verbose=True)
    
    # Bypass 모드면 simple_embedding gradient 확인
    if BYPASS_GNN:
        simple_grads = [v['norm'] for k, v in grad_info.items() if 'simple_embedding' in k]
        if simple_grads and sum(simple_grads) > 0:
            print("✅ Simple Embedding Gradient 흐름 확인됨")
        else:
            print("❌ Simple Embedding Gradient 없음!")
    else:
        gnn_grads = module_grads.get('gnn_encoder', [])
        if gnn_grads and sum(gnn_grads) > 0:
            print("✅ GNN Gradient 흐름 확인됨")
        else:
            print("❌ GNN Gradient 없음!")
    
    optimizer.zero_grad()  # 테스트 후 초기화
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 🔥 Forward (Teacher Forcing 100% 강제)
        # 첫 에폭에서만 debug 출력
        predictions = model(batch, training=True, teacher_forcing_ratio=1.0, debug=(epoch == 0))
        
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
            grad_info, module_grads = check_gradients(model, verbose=(epoch == 0))
            nan_grads = [k for k, v in grad_info.items() if v['has_nan']]
            inf_grads = [k for k, v in grad_info.items() if v['has_inf']]
            zero_grads = [k for k, v in grad_info.items() if v['norm'] < 1e-7]
            
            if nan_grads:
                print(f"⚠️ NaN Gradients: {nan_grads}")
            if inf_grads:
                print(f"⚠️ Inf Gradients: {inf_grads}")
            if zero_grads and epoch == 0:
                print(f"⚠️ Zero Gradients ({len(zero_grads)}개): {zero_grads[:3]}...")
                
            # 🔥 GNN gradient 특별 확인
            gnn_grads = module_grads.get('gnn_encoder', [])
            if gnn_grads and max(gnn_grads) < 1e-7:
                print(f"❌ GNN Gradient 끊김 감지! 모든 GNN param의 grad가 0")
            elif gnn_grads:
                print(f"✅ GNN Gradient 정상: avg={sum(gnn_grads)/len(gnn_grads):.6f}")
        
        # 🔥 Gradient clipping (max_norm=1.0으로 타이트하게 - Explosion 방지)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Gradient norm 체크
        if epoch == 0:
            if grad_norm < 1e-6:
                print(f"⚠️ Gradient norm이 매우 작음: {grad_norm:.2e}")
            else:
                print(f"✅ 초기 Gradient norm: {grad_norm:.4f}")
        
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
        max_n = batch['actual_route'].size(1)
        actual = batch['actual_route'][i, :n].cpu().numpy()
        predicted = predictions['routes'][i, :n].cpu().numpy()
        
        # 🔥 패딩 마스킹 검증: 예측값이 유효 범위 내인지 확인
        invalid_preds = [p for p in predicted if p >= n]
        
        match = (actual == predicted).sum()
        
        print(f"\n샘플 {i} (유효 노드 {n}개, 패딩 포함 {max_n}개):")
        print(f"   정답 경로:    {actual}")
        print(f"   예측 경로:    {predicted}")
        print(f"   일치율:       {match}/{n} ({match/n*100:.1f}%)")
        
        if invalid_preds:
            print(f"   ❌ 패딩 마스킹 실패! 잘못된 인덱스: {invalid_preds}")
        else:
            print(f"   ✅ 패딩 마스킹 정상 (모든 예측이 0~{n-1} 범위)")
    
    return history


if __name__ == "__main__":
    history = overfit_test(
        data_path='./model_data/train_data.pkl',
        num_samples=10,
        num_epochs=200,       # 200 에폭
        learning_rate=0.001,  # 🔥 기본 LR (x10 부스트 → 0.01)
        print_every=5         # 🔥 5 에폭마다 출력 (변화 관찰)
    )
