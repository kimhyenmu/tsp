import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import wandb  # optional
from pathlib import Path
import json
from dataset_loss import DeliveryDataset, collate_fn, HybridLoss, EvaluationMetrics
from model.topmodel import HybridRoutingModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
# ==========================================
# 6. Trainer 클래스
# ==========================================
class Trainer:
    """
    모델 학습 관리 클래스
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # 저장 디렉토리
        self.save_dir = Path(config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 최고 성능 추적
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # 학습 히스토리
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_route_acc': [],
            'val_route_acc': [],
            'train_time_mae': [],
            'val_time_mae': []
        }
        
        # WandB 초기화 (optional)
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'delivery-routing'),
                config=config
            )
    
    def train_epoch(self, epoch):
        """1 에폭 학습"""
        self.model.train()
        
        total_loss = 0
        total_route_loss = 0
        total_time_loss = 0
        total_route_acc = 0
        total_time_mae = 0
        num_batches = 0
        
        # 🔥 개선된 Teacher forcing 스케줄
        # - 초반 20 에폭: 100% teacher forcing (안정적 학습)
        # - 이후 점진적 감소 (Curriculum Learning)
        if epoch < 20:
            teacher_forcing_ratio = 1.0
        else:
            # 20 에폭 이후부터 천천히 감소 (0.98^(epoch-20))
            teacher_forcing_ratio = max(1.0 * (0.98 ** (epoch - 20)), 0.3)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            # GPU로 이동
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward
            self.optimizer.zero_grad()
            
            predictions = self.model(
                batch, 
                training=True, 
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            
            # Loss 계산
            targets = {
                'actual_route': batch['actual_route'],
                'actual_times': batch['actual_times']
            }
            
            loss_dict = self.criterion(predictions, targets, batch['num_nodes'])
            
            loss = loss_dict['total_loss']
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            total_route_loss += loss_dict['route_loss'].item()
            total_time_loss += loss_dict['time_loss'].item()
            total_route_acc += loss_dict['route_accuracy']
            total_time_mae += loss_dict['time_mae'].item()
            num_batches += 1
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'route_acc': f"{loss_dict['route_accuracy']:.3f}",
                'time_mae': f"{loss_dict['time_mae'].item():.1f}s"
            })
        
        # 평균 계산
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_route_loss': total_route_loss / num_batches,
            'train_time_loss': total_time_loss / num_batches,
            'train_route_acc': total_route_acc / num_batches,
            'train_time_mae': total_time_mae / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """검증"""
        self.model.eval()
        
        total_loss = 0
        total_route_loss = 0
        total_time_loss = 0
        total_route_acc = 0
        total_time_mae = 0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        all_num_nodes = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward (no teacher forcing - 실제 성능 평가)
            predictions = self.model(batch, training=False)

            targets = {
                'actual_route': batch['actual_route'],
                'actual_times': batch['actual_times']
            }
            
            loss_dict = self.criterion(predictions, targets, batch['num_nodes'])
            
            total_loss += loss_dict['total_loss'].item()
            total_route_loss += loss_dict['route_loss'].item()
            total_time_loss += loss_dict['time_loss'].item()
            total_route_acc += loss_dict['route_accuracy']
            total_time_mae += loss_dict['time_mae'].item()
            num_batches += 1
            
            # 상세 평가용 저장
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_num_nodes.extend(batch['num_nodes'])
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'route_acc': f"{loss_dict['route_accuracy']:.3f}"
            })
        
        # 평균 메트릭
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_route_loss': total_route_loss / num_batches,
            'val_time_loss': total_time_loss / num_batches,
            'val_route_acc': total_route_acc / num_batches,
            'val_time_mae': total_time_mae / num_batches
        }
        
        # 상세 평가 (첫 배치만)
        if len(all_predictions) > 0:
            detailed_metrics = EvaluationMetrics.compute_all_metrics(
                all_predictions[0], 
                all_targets[0], 
                batch['num_nodes']
            )
            metrics.update({f'val_{k}': v for k, v in detailed_metrics.items()})
        
        return metrics
    
    def train(self, num_epochs):
        """전체 학습 루프"""
        print("=" * 60)
        print("🚀 학습 시작")
        print("=" * 60)
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # 학습
            train_metrics = self.train_epoch(epoch)
            
            # 검증
            val_metrics = self.validate(epoch)
            
            # 스케줄러 업데이트
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # 히스토리 업데이트
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['train_route_acc'].append(train_metrics['train_route_acc'])
            self.history['val_route_acc'].append(val_metrics['val_route_acc'])
            self.history['train_time_mae'].append(train_metrics['train_time_mae'])
            self.history['val_time_mae'].append(val_metrics['val_time_mae'])
            
            # 결과 출력
            print(f"\n📊 Epoch {epoch + 1} 결과:")
            print(f"   Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"   Train Route Acc: {train_metrics['train_route_acc']:.3f} | Val Route Acc: {val_metrics['val_route_acc']:.3f}")
            print(f"   Train Time MAE: {train_metrics['train_time_mae']:.1f}s | Val Time MAE: {val_metrics['val_time_mae']:.1f}s")
            
            if 'val_kendall_tau' in val_metrics:
                print(f"   Val Kendall's Tau: {val_metrics['val_kendall_tau']:.3f}")
                print(f"   Val Total Time Error: {val_metrics['val_total_time_error']:.2f}%")
            
            # WandB 로깅
            if self.config.get('use_wandb', False):
                wandb.log({**train_metrics, **val_metrics, 'epoch': epoch})
            
            # 최고 모델 저장
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"   ✅ 최고 모델 저장! (Val Loss: {val_metrics['val_loss']:.4f})")
            
            # 주기적 체크포인트
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        print("\n" + "=" * 60)
        print("🎉 학습 완료!")
        print(f"   최고 성능: Epoch {self.best_epoch + 1}, Val Loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # 히스토리 저장
        self.save_history()
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """학습 히스토리 저장"""
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"📁 학습 히스토리 저장: {self.save_dir / 'history.json'}")


# ==========================================
# 7. 메인 학습 스크립트
# ==========================================
def main():
    # ==========================================
    # 설정
    # ==========================================
    config = {
        # 데이터
        'train_data_path': './model_data/train_data.pkl',
        'val_data_path': './model_data/val_data.pkl',
        'batch_size': 32,
        'num_workers': 4,
        
        # 모델
        'node_dim': 2,
        'hidden_dim': 128,
        'gnn_layers': 3,
        'gnn_heads': 4,
        'tf_layers': 4,
        'tf_heads': 8,
        'tf_ff_dim': 512,
        'pointer_heads': 4,
        'dropout': 0.1,
        'k_neighbors': 10,
        
        # 학습
        'num_epochs': 100,
        'learning_rate': 3e-4,  # 🔥 안정적인 학습률
        'weight_decay': 1e-5,
        'route_weight': 1.0,
        'time_weight': 0.05,    # 🔥 Time weight (로그 스케일 기준)
        'label_smoothing': 0.0, # 🔥 마스킹과 충돌 방지
        
        # 스케줄러
        'scheduler_type': 'plateau',  # 'plateau' or 'cosine'
        'patience': 5,
        
        # 저장
        'save_dir': './checkpoints',
        'save_every': 10,
        
        # WandB (optional)
        'use_wandb': False,
        'wandb_project': 'delivery-routing'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # ==========================================
    # 데이터 로더
    # ==========================================
    print("\n📂 데이터 로드 중...")
    
    train_dataset = DeliveryDataset(config['train_data_path'])
    val_dataset = DeliveryDataset(config['val_data_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # ==========================================
    # 모델 초기화
    # ==========================================
    print("\n🧠 모델 초기화 중...")
    
      # 이전 파일에서 import
    model = HybridRoutingModel(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        gnn_layers=config['gnn_layers'],
        gnn_heads=config['gnn_heads'],
        tf_layers=config['tf_layers'],
        tf_heads=config['tf_heads'],
        tf_ff_dim=config['tf_ff_dim'],
        pointer_heads=config['pointer_heads'],
        dropout=config['dropout'],
        k_neighbors=config['k_neighbors']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - 총 파라미터: {total_params:,}")
    print(f"   - 학습 가능 파라미터: {trainable_params:,}")
    print(f"   - 메모리: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # ==========================================
    # Loss & Optimizer
    # ==========================================
    criterion = HybridLoss(
        route_weight=config['route_weight'],
        time_weight=config['time_weight'],
        smoothing=config['label_smoothing']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    if config['scheduler_type'] == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config['patience']
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )
    
    # ==========================================
    # Trainer 초기화 및 학습
    # ==========================================
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # 학습 시작
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()