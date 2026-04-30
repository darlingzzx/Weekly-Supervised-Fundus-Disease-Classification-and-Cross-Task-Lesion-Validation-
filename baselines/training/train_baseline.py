"""
基线训练脚本 - 统一训练所有基线方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from baselines.models.baselines import create_baseline


class BaselineTrainer:
    """基线训练器"""
    
    def __init__(
        self,
        model,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        save_dir: str = 'baselines/results'
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
        # 训练历史
        self.train_history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        task: str = 'odir'
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            task: 任务类型 ('odir' 或 'ddr')
        
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 选择损失函数
        if task == 'odir':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # TODO: 根据你的数据加载器调整
            # images = batch['image'].to(self.device)
            # labels = batch['label'].to(self.device)
            
            # 示例数据（实际使用时删除）
            images = torch.randn(batch.get('image', torch.randn(16, 3, 224, 224)).shape).to(self.device)
            if task == 'odir':
                labels = torch.randint(0, 2, (images.shape[0], 8)).float().to(self.device)
            else:
                labels = torch.randint(0, 5, (images.shape[0],)).to(self.device)
            
            # 前向传播
            if task == 'multi_task':
                odir_logits, ddr_logits = self.model(images, task='both')
                loss = criterion(odir_logits, labels[:, :8]) + criterion(ddr_logits, labels[:, 8].long())
            else:
                logits = self.model(images, task=task)
                loss = criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            if task == 'odir':
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct += (predictions == labels).all(dim=1).sum().item()
                total += labels.shape[0]
            elif task == 'ddr':
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.shape[0]
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        task: str = 'odir'
    ) -> Dict[str, float]:
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
            task: 任务类型
        
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        if task == 'odir':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # TODO: 根据你的数据加载器调整
                # images = batch['image'].to(self.device)
                # labels = batch['label'].to(self.device)
                
                # 示例数据（实际使用时删除）
                images = torch.randn(batch.get('image', torch.randn(16, 3, 224, 224)).shape).to(self.device)
                if task == 'odir':
                    labels = torch.randint(0, 2, (images.shape[0], 8)).float().to(self.device)
                else:
                    labels = torch.randint(0, 5, (images.shape[0],)).to(self.device)
                
                # 前向传播
                if task == 'multi_task':
                    odir_logits, ddr_logits = self.model(images, task='both')
                    loss = criterion(odir_logits, labels[:, :8]) + criterion(ddr_logits, labels[:, 8].long())
                else:
                    logits = self.model(images, task=task)
                    loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                if task == 'odir':
                    predictions = (torch.sigmoid(logits) > 0.5).float()
                    correct += (predictions == labels).all(dim=1).sum().item()
                    total += labels.shape[0]
                elif task == 'ddr':
                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.shape[0]
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        task: str = 'odir',
        save_checkpoint: bool = True
    ):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            task: 任务类型
            save_checkpoint: 是否保存检查点
        """
        print(f"开始训练 {self.model.model_name} ({task}任务)")
        print(f"设备: {self.device}")
        print(f"训练轮数: {num_epochs}")
        
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, task)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            
            # 验证
            val_metrics = self.validate(val_loader, task)
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            })
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_accuracy:
                best_accuracy = val_metrics['accuracy']
                if save_checkpoint:
                    checkpoint_path = self.save_dir / f"{self.model.model_name}_{task}_best.pth"
                    self.model.save_checkpoint(str(checkpoint_path))
                    print(f"保存最佳模型: {checkpoint_path}")
        
        # 保存训练历史
        history_path = self.save_dir / f"{self.model.model_name}_{task}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"训练完成！最佳验证准确率: {best_accuracy:.4f}")
        return best_accuracy


def train_baseline(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    task: str = 'odir',
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = 'baselines/results'
):
    """
    训练单个基线模型
    
    Args:
        model_name: 模型名称
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        task: 任务类型
        learning_rate: 学习率
        device: 设备
        save_dir: 保存目录
    
    Returns:
        best_accuracy: 最佳验证准确率
    """
    # 创建模型
    model = create_baseline(model_name)
    
    # 创建训练器
    trainer = BaselineTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    # 训练
    best_accuracy = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        task=task
    )
    
    return best_accuracy


if __name__ == "__main__":
    # TODO: 导入你的数据加载器
    # from your_dataloader import ODIRDataset, DDRDataset
    
    # TODO: 创建数据加载器
    # train_dataset = ODIRDataset('data/final_split/ODIR/train')
    # val_dataset = ODIRDataset('data/final_split/ODIR/val')
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 示例：训练ResNet50基线
    print("示例：训练ResNet50基线")
    print("TODO: 请先实现数据加载器，然后取消注释以下代码")
    
    # best_acc = train_baseline(
    #     model_name='resnet50',
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=100,
    #     task='odir',
    #     learning_rate=1e-4
    # )
    
    # print(f"ResNet50最佳准确率: {best_acc:.4f}")