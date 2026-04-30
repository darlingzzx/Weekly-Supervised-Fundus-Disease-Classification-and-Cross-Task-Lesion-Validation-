"""
顶级可视化脚本 - 生成论文级别的可视化图表

包含：
1. 性能对比图表（柱状图、折线图、雷达图）
2. 消融实验可视化
3. 训练曲线可视化
4. 注意力热图可视化
5. 混淆矩阵可视化
6. 统计显著性可视化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 设置颜色方案
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'warning': '#3B1F2B',
    'info': '#06A77D',
    'light': '#F0F4F8',
    'dark': '#1A1A2E'
}


class TopLevelVisualizer:
    """顶级可视化器"""
    
    def __init__(self, output_dir: str = 'baselines/visualization/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        self.set_style()
    
    def set_style(self):
        """设置可视化样式"""
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_performance_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['mAP', 'macro_F1', 'macro_AUC'],
        title: str = 'Performance Comparison',
        save_name: str = 'performance_comparison'
    ):
        """
        绘制性能对比柱状图
        
        Args:
            results: 各模型的结果字典
            metrics: 要对比的指标
            title: 图表标题
            save_name: 保存文件名
        """
        # 准备数据
        models = list(results.keys())
        data = []
        
        for model in models:
            for metric in metrics:
                value = results[model].get(metric, 0.0)
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制柱状图
        sns.barplot(
            data=df,
            x='Model',
            y='Value',
            hue='Metric',
            palette='husl',
            ax=ax
        )
        
        # 设置样式
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"性能对比图已保存: {save_path}")
    
    def plot_ablation_study(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['mAP', 'macro_F1'],
        title: str = 'Ablation Study',
        save_name: str = 'ablation_study'
    ):
        """
        绘制消融实验结果
        
        Args:
            ablation_results: 消融实验结果
            metrics: 要对比的指标
            title: 图表标题
            save_name: 保存文件名
        """
        # 准备数据
        configs = list(ablation_results.keys())
        x = np.arange(len(configs))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制每个指标
        for i, metric in enumerate(metrics):
            values = [ablation_results[config].get(metric, 0.0) for config in configs]
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        # 设置样式
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"消融实验图已保存: {save_path}")
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = 'Training Curves',
        save_name: str = 'training_curves'
    ):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史字典
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        if 'train_loss' in history and 'val_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Loss', fontsize=11, fontweight='bold')
            axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # 准确率曲线
        if 'train_accuracy' in history and 'val_accuracy' in history:
            axes[1].plot(history['train_accuracy'], label='Train Acc', linewidth=2)
            axes[1].plot(history['val_accuracy'], label='Val Acc', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
            axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"训练曲线图已保存: {save_path}")
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        title: str = 'Confusion Matrix',
        save_name: str = 'confusion_matrix'
    ):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热图
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"混淆矩阵图已保存: {save_path}")
    
    def plot_radar_chart(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        title: str = 'Radar Chart Comparison',
        save_name: str = 'radar_chart'
    ):
        """
        绘制雷达图对比
        
        Args:
            results: 各模型的结果
            metrics: 要对比的指标
            title: 图表标题
            save_name: 保存文件名
        """
        # 准备数据
        models = list(results.keys())
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 绘制每个模型
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            values = [results[model].get(metric, 0.0) for metric in metrics]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # 设置样式
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"雷达图已保存: {save_path}")
    
    def plot_attention_heatmap(
        self,
        attention_map: np.ndarray,
        original_image: np.ndarray,
        title: str = 'Attention Heatmap',
        save_name: str = 'attention_heatmap'
    ):
        """
        绘制注意力热图
        
        Args:
            attention_map: 注意力图
            original_image: 原始图像
            title: 图表标题
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 注意力图
        im1 = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('Attention Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 叠加图
        axes[2].imshow(original_image)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"注意力热图已保存: {save_path}")
    
    def plot_significance_test(
        self,
        baseline_results: Dict[str, float],
        our_results: Dict[str, float],
        p_values: Dict[str, float],
        metrics: List[str],
        title: str = 'Statistical Significance Test',
        save_name: str = 'significance_test'
    ):
        """
        绘制统计显著性检验结果
        
        Args:
            baseline_results: 基线方法结果
            our_results: 我们的方法结果
            p_values: p值
            metrics: 指标列表
            title: 图表标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # 绘制柱状图
        baseline_values = [baseline_results.get(m, 0.0) for m in metrics]
        our_values = [our_results.get(m, 0.0) for m in metrics]
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        bars2 = ax.bar(x + width/2, our_values, width, label='Ours', alpha=0.8)
        
        # 添加显著性标记
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            p_val = p_values.get(metrics[i], 1.0)
            
            # 绘制显著性线
            y_max = max(height1, height2)
            y_offset = 0.02
            
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = 'ns'
            
            ax.text(i, y_max + y_offset, marker, 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # 设置样式
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 添加图例说明
        legend_elements = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='*** p < 0.001'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='** p < 0.01'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='* p < 0.05'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='ns not significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"显著性检验图已保存: {save_path}")
    
    def create_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        save_name: str = 'comparison_table'
    ):
        """
        创建对比表格
        
        Args:
            results: 各模型结果
            metrics: 指标列表
            save_name: 保存文件名
        """
        # 准备数据
        models = list(results.keys())
        data = []
        
        for metric in metrics:
            row = {'Metric': metric}
            for model in models:
                value = results[model].get(metric, 0.0)
                row[model] = f"{value:.4f}"
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # 绘制表格
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f3f3f3'] * len(df.columns),
            colWidths=[0.15] + [0.15] * len(models)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 设置表头样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置最佳结果高亮
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0.0) for model in models]
            max_value = max(values)
            for j, model in enumerate(models):
                if results[model].get(metric, 0.0) == max_value:
                    table[(i+1, j+1)].set_facecolor('#C73E1D')
                    table[(i+1, j+1)].set_text_props(weight='bold', color='white')
        
        plt.title('Performance Comparison Table', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # 保存
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print(f"对比表格已保存: {save_path}")


def load_results_from_json(json_files: List[str]) -> Dict[str, Dict[str, float]]:
    """
    从JSON文件加载结果
    
    Args:
        json_files: JSON文件路径列表
    
    Returns:
        results: 结果字典
    """
    results = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            model_name = data.get('model_name', 'unknown')
            metrics = data.get('metrics', {})
            results[model_name] = metrics
    
    return results


if __name__ == "__main__":
    # 创建可视化器
    visualizer = TopLevelVisualizer()
    
    # 示例数据
    example_results = {
        'ResNet50': {'mAP': 0.82, 'macro_F1': 0.80, 'macro_AUC': 0.85},
        'DenseNet121': {'mAP': 0.83, 'macro_F1': 0.81, 'macro_AUC': 0.86},
        'ViT-B': {'mAP': 0.84, 'macro_F1': 0.82, 'macro_AUC': 0.87},
        'Ours': {'mAP': 0.88, 'macro_F1': 0.86, 'macro_AUC': 0.90}
    }
    
    # 示例：绘制性能对比图
    print("示例：绘制性能对比图")
    visualizer.plot_performance_comparison(
        results=example_results,
        metrics=['mAP', 'macro_F1', 'macro_AUC'],
        title='Performance Comparison on ODIR Dataset'
    )
    
    # 示例：绘制雷达图
    print("示例：绘制雷达图")
    visualizer.plot_radar_chart(
        results=example_results,
        metrics=['mAP', 'macro_F1', 'macro_AUC'],
        title='Multi-Metric Comparison'
    )
    
    # 示例：创建对比表格
    print("示例：创建对比表格")
    visualizer.create_comparison_table(
        results=example_results,
        metrics=['mAP', 'macro_F1', 'macro_AUC']
    )
    
    print("\n示例可视化完成！")
    print("TODO: 使用实际数据替换示例数据")