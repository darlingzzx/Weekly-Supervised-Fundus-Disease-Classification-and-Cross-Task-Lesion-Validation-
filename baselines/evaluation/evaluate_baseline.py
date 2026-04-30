"""
基线评估脚本 - 统一评估所有基线方法
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, List
from tqdm import tqdm
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent.parent))

from baselines.models.baselines import create_baseline


class BaselineEvaluator:
    """基线评估器"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_odir(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        评估ODIR多标签分类
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 评估指标
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating ODIR'):
                # TODO: 根据你的数据加载器调整
                # images = batch['image'].to(self.device)
                # labels = batch['label'].to(self.device)
                
                # 示例数据（实际使用时删除）
                images = torch.randn(batch.get('image', torch.randn(16, 3, 224, 224)).shape).to(self.device)
                labels = torch.randint(0, 2, (images.shape[0], 8)).float().to(self.device)
                
                # 前向传播
                logits = self.model(images, task='odir')
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probabilities = torch.cat(all_probabilities).numpy()
        
        # 计算指标
        metrics = {}
        
        # mAP
        try:
            mAP = average_precision_score(all_labels, all_probabilities, average='macro')
            metrics['mAP'] = mAP
        except:
            metrics['mAP'] = 0.0
        
        # AUC
        try:
            auc_scores = []
            for i in range(all_labels.shape[1]):
                if all_labels[:, i].sum() > 0:
                    auc = roc_auc_score(all_labels[:, i], all_probabilities[:, i])
                    auc_scores.append(auc)
            metrics['macro_AUC'] = np.mean(auc_scores) if auc_scores else 0.0
            metrics['micro_AUC'] = roc_auc_score(all_labels.ravel(), all_probabilities.ravel())
        except:
            metrics['macro_AUC'] = 0.0
            metrics['micro_AUC'] = 0.0
        
        # F1, Precision, Recall
        metrics['macro_F1'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_F1'] = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        metrics['macro_Precision'] = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_Precision'] = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
        metrics['macro_Recall'] = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_Recall'] = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
        
        # 样本准确率
        sample_accuracy = (all_predictions == all_labels).all(axis=1).mean()
        metrics['sample_accuracy'] = sample_accuracy
        
        return metrics
    
    def evaluate_ddr(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        评估DDR分级分类
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 评估指标
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating DDR'):
                # TODO: 根据你的数据加载器调整
                # images = batch['image'].to(self.device)
                # labels = batch['label'].to(self.device)
                
                # 示例数据（实际使用时删除）
                images = torch.randn(batch.get('image', torch.randn(16, 3, 224, 224)).shape).to(self.device)
                labels = torch.randint(0, 5, (images.shape[0],)).to(self.device)
                
                # 前向传播
                logits = self.model(images, task='ddr')
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(probabilities.cpu())
        
        # 合并所有批次
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probabilities = torch.cat(all_probabilities).numpy()
        
        # 计算指标
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(all_labels, all_predictions)
        
        # Kappa
        try:
            metrics['quadratic_kappa'] = cohen_kappa_score(
                all_labels, all_predictions, weights='quadratic'
            )
            metrics['linear_kappa'] = cohen_kappa_score(
                all_labels, all_predictions, weights='linear'
            )
        except:
            metrics['quadratic_kappa'] = 0.0
            metrics['linear_kappa'] = 0.0
        
        # F1, Precision, Recall
        metrics['macro_F1'] = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_F1'] = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        metrics['macro_Precision'] = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_Precision'] = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
        metrics['macro_Recall'] = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        metrics['micro_Recall'] = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_predictions).tolist()
        
        return metrics
    
    def evaluate_multi_task(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        评估多任务模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 评估指标
        """
        odir_predictions = []
        odir_labels = []
        ddr_predictions = []
        ddr_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating Multi-Task'):
                # TODO: 根据你的数据加载器调整
                # images = batch['image'].to(self.device)
                # odir_labels_batch = batch['odir_label'].to(self.device)
                # ddr_labels_batch = batch['ddr_label'].to(self.device)
                
                # 示例数据（实际使用时删除）
                images = torch.randn(batch.get('image', torch.randn(16, 3, 224, 224)).shape).to(self.device)
                odir_labels_batch = torch.randint(0, 2, (images.shape[0], 8)).float().to(self.device)
                ddr_labels_batch = torch.randint(0, 5, (images.shape[0],)).to(self.device)
                
                # 前向传播
                odir_logits, ddr_logits = self.model(images, task='both')
                
                odir_probs = torch.sigmoid(odir_logits)
                odir_pred = (odir_probs > 0.5).float()
                
                ddr_pred = torch.argmax(ddr_logits, dim=1)
                
                odir_predictions.append(odir_pred.cpu())
                odir_labels.append(odir_labels_batch.cpu())
                ddr_predictions.append(ddr_pred.cpu())
                ddr_labels.append(ddr_labels_batch.cpu())
        
        # 合并所有批次
        odir_predictions = torch.cat(odir_predictions).numpy()
        odir_labels = torch.cat(odir_labels).numpy()
        ddr_predictions = torch.cat(ddr_predictions).numpy()
        ddr_labels = torch.cat(ddr_labels).numpy()
        
        # 计算ODIR指标
        odir_metrics = {
            'ODIR_sample_accuracy': (odir_predictions == odir_labels).all(axis=1).mean(),
            'ODIR_macro_F1': f1_score(odir_labels, odir_predictions, average='macro', zero_division=0),
            'ODIR_micro_F1': f1_score(odir_labels, odir_predictions, average='micro', zero_division=0)
        }
        
        # 计算DDR指标
        ddr_metrics = {
            'DDR_accuracy': accuracy_score(ddr_labels, ddr_predictions),
            'DDR_quadratic_kappa': cohen_kappa_score(ddr_labels, ddr_predictions, weights='quadratic'),
            'DDR_macro_F1': f1_score(ddr_labels, ddr_predictions, average='macro', zero_division=0)
        }
        
        # 合并指标
        metrics = {**odir_metrics, **ddr_metrics}
        
        return metrics


def evaluate_baseline(
    model_name: str,
    test_loader: DataLoader,
    task: str = 'odir',
    checkpoint_path: str = None,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估单个基线模型
    
    Args:
        model_name: 模型名称
        test_loader: 测试数据加载器
        task: 任务类型
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        metrics: 评估指标
    """
    # 创建模型
    model = create_baseline(
        model_name=model_name,
        checkpoint_path=checkpoint_path
    )
    
    # 创建评估器
    evaluator = BaselineEvaluator(model, device=device)
    
    # 评估
    if task == 'odir':
        metrics = evaluator.evaluate_odir(test_loader)
    elif task == 'ddr':
        metrics = evaluator.evaluate_ddr(test_loader)
    elif task == 'multi_task':
        metrics = evaluator.evaluate_multi_task(test_loader)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return metrics


def save_evaluation_results(
    model_name: str,
    task: str,
    metrics: Dict[str, float],
    save_dir: str = 'baselines/results'
):
    """
    保存评估结果
    
    Args:
        model_name: 模型名称
        task: 任务类型
        metrics: 评估指标
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    result_file = save_path / f"{model_name}_{task}_results.json"
    
    result_data = {
        'model_name': model_name,
        'task': task,
        'metrics': metrics
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"评估结果已保存到: {result_file}")


if __name__ == "__main__":
    # TODO: 导入你的数据加载器
    # from your_dataloader import ODIRDataset, DDRDataset
    
    # TODO: 创建测试数据加载器
    # test_dataset = ODIRDataset('data/final_split/ODIR/test')
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 示例：评估ResNet50基线
    print("示例：评估ResNet50基线")
    print("TODO: 请先实现数据加载器，然后取消注释以下代码")
    
    # metrics = evaluate_baseline(
    #     model_name='resnet50',
    #     test_loader=test_loader,
    #     task='odir',
    #     checkpoint_path='baselines/results/resnet50_odir_best.pth'
    # )
    
    # save_evaluation_results('resnet50', 'odir', metrics)
    
    # print("ResNet50评估结果:")
    # for key, value in metrics.items():
    #     print(f"  {key}: {value:.4f}")