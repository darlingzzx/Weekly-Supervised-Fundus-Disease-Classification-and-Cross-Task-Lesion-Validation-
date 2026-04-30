# 基线对比框架

## 📁 目录结构

```
baselines/
├── models/                    # 模型定义
│   └── baselines.py          # 所有基线模型
├── training/                  # 训练脚本
│   └── train_baseline.py     # 统一训练脚本
├── evaluation/                # 评估脚本
│   └── evaluate_baseline.py  # 统一评估脚本
├── visualization/             # 可视化脚本
│   └── visualize.py         # 顶级可视化
├── configs/                  # 配置文件
│   ├── baseline_config.yaml  # 基线配置
│   └── model_config.yaml    # 模型配置
├── results/                  # 结果保存
│   ├── checkpoints/         # 模型检查点
│   ├── metrics/            # 评估指标
│   └── figures/           # 可视化图表
└── README.md               # 使用说明
```

## 🎯 支持的基线方法

### CNN基线
- ✅ ResNet50
- ✅ DenseNet121
- ✅ EfficientNet-B3

### Transformer基线
- ✅ ViT-B (ViT-Base)
- ✅ ViT-L (ViT-Large)
- ✅ Swin-T (Swin-Tiny)
- ✅ Swin-S (Swin-Small)

### 预训练基线
- ✅ MAE预训练
- ✅ DINO预训练
- ✅ RETFound预训练

### 多任务基线
- ✅ 简单多任务学习
- ✅ 用户自定义模型

## 🚀 快速开始

### 1. 训练单个基线

```python
from baselines.training.train_baseline import train_baseline
from torch.utils.data import DataLoader

# TODO: 导入你的数据加载器
# from your_dataloader import ODIRDataset

# TODO: 创建数据加载器
# train_dataset = ODIRDataset('data/final_split/ODIR/train')
# val_dataset = ODIRDataset('data/final_split/ODIR/val')
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 训练ResNet50基线
best_acc = train_baseline(
    model_name='resnet50',
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    task='odir',
    learning_rate=1e-4,
    device='cuda',
    save_dir='baselines/results'
)

print(f"ResNet50最佳准确率: {best_acc:.4f}")
```

### 2. 评估训练好的模型

```python
from baselines.evaluation.evaluate_baseline import evaluate_baseline, save_evaluation_results

# TODO: 创建测试数据加载器
# test_dataset = ODIRDataset('data/final_split/ODIR/test')
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 评估模型
metrics = evaluate_baseline(
    model_name='resnet50',
    test_loader=test_loader,
    task='odir',
    checkpoint_path='baselines/results/resnet50_odir_best.pth',
    device='cuda'
)

# 保存结果
save_evaluation_results('resnet50', 'odir', metrics)

print("评估结果:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")
```

### 3. 批量训练所有基线

```python
from baselines.training.train_baseline import train_baseline

# 基线模型列表
baseline_models = [
    'resnet50',
    'densenet121',
    'efficientnet_b3',
    'vit_b',
    'swin_t'
]

# 批量训练
for model_name in baseline_models:
    print(f"\n训练 {model_name}...")
    
    best_acc = train_baseline(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        task='odir'
    )
    
    print(f"{model_name} 最佳准确率: {best_acc:.4f}")
```

### 4. 批量评估所有基线

```python
from baselines.evaluation.evaluate_baseline import evaluate_baseline
import json

# 评估所有基线
all_results = {}

for model_name in baseline_models:
    print(f"\n评估 {model_name}...")
    
    metrics = evaluate_baseline(
        model_name=model_name,
        test_loader=test_loader,
        task='odir',
        checkpoint_path=f'baselines/results/{model_name}_odir_best.pth'
    )
    
    all_results[model_name] = metrics
    
    # 保存结果
    save_evaluation_results(model_name, 'odir', metrics)

# 保存所有结果
with open('baselines/results/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("所有基线评估完成！")
```

### 5. 生成顶级可视化

```python
from baselines.visualization.visualize import TopLevelVisualizer, load_results_from_json

# 创建可视化器
visualizer = TopLevelVisualizer()

# 加载所有结果
json_files = [
    'baselines/results/resnet50_odir_results.json',
    'baselines/results/densenet121_odir_results.json',
    'baselines/results/vit_b_odir_results.json',
    'baselines/results/our_method_odir_results.json'
]

results = load_results_from_json(json_files)

# 绘制性能对比图
visualizer.plot_performance_comparison(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    title='Performance Comparison on ODIR Dataset',
    save_name='performance_comparison'
)

# 绘制雷达图
visualizer.plot_radar_chart(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    title='Multi-Metric Comparison',
    save_name='radar_chart'
)

# 创建对比表格
visualizer.create_comparison_table(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    save_name='comparison_table'
)
```

## 📊 支持的可视化类型

### 1. 性能对比图
```python
visualizer.plot_performance_comparison(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    title='Performance Comparison'
)
```

### 2. 消融实验图
```python
visualizer.plot_ablation_study(
    ablation_results=ablation_results,
    metrics=['mAP', 'macro_F1'],
    title='Ablation Study'
)
```

### 3. 训练曲线图
```python
visualizer.plot_training_curves(
    history=training_history,
    title='Training Curves'
)
```

### 4. 混淆矩阵
```python
visualizer.plot_confusion_matrix(
    confusion_matrix=conf_mat,
    class_names=['Class 0', 'Class 1', 'Class 2'],
    title='Confusion Matrix'
)
```

### 5. 雷达图
```python
visualizer.plot_radar_chart(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    title='Radar Chart Comparison'
)
```

### 6. 注意力热图
```python
visualizer.plot_attention_heatmap(
    attention_map=attention,
    original_image=image,
    title='Attention Heatmap'
)
```

### 7. 统计显著性检验
```python
visualizer.plot_significance_test(
    baseline_results=baseline_metrics,
    our_results=our_metrics,
    p_values=p_values,
    metrics=['mAP', 'macro_F1', 'macro_AUC'],
    title='Statistical Significance Test'
)
```

### 8. 对比表格
```python
visualizer.create_comparison_table(
    results=results,
    metrics=['mAP', 'macro_F1', 'macro_AUC']
)
```

## 🎨 可视化特点

### 顶级质量
- ✅ 300 DPI高分辨率
- ✅ 论文级图表样式
- ✅ 专业配色方案
- ✅ 清晰的图例和标签

### 自定义样式
- ✅ 支持中文字体
- ✅ 可调整图表大小
- ✅ 可自定义颜色方案
- ✅ 支持多种图表类型

### 自动化处理
- ✅ 自动保存高分辨率图片
- ✅ 自动调整布局
- ✅ 自动添加网格和图例
- ✅ 自动高亮最佳结果

## 🔧 配置说明

### 模型配置
```python
# 创建基线模型
model = create_baseline(
    model_name='resnet50',           # 模型名称
    num_classes_odir=8,              # ODIR类别数
    num_classes_ddr=5,               # DDR类别数
    pretrained=True,                  # 是否使用预训练权重
    checkpoint_path=None              # 检查点路径
)
```

### 训练配置
```python
# 训练参数
train_baseline(
    model_name='resnet50',
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,                  # 训练轮数
    task='odir',                    # 任务类型
    learning_rate=1e-4,             # 学习率
    device='cuda',                   # 设备
    save_dir='baselines/results'     # 保存目录
)
```

### 评估配置
```python
# 评估参数
evaluate_baseline(
    model_name='resnet50',
    test_loader=test_loader,
    task='odir',                    # 任务类型
    checkpoint_path='path/to/checkpoint.pth',
    device='cuda'                    # 设备
)
```

## 📋 评估指标

### ODIR多标签分类
- **mAP**: 平均精度均值
- **macro_AUC**: 宏平均AUC
- **micro_AUC**: 微平均AUC
- **macro_F1**: 宏平均F1分数
- **micro_F1**: 微平均F1分数
- **macro_Precision**: 宏平均精确率
- **micro_Precision**: 微平均精确率
- **macro_Recall**: 宏平均召回率
- **micro_Recall**: 微平均召回率
- **sample_accuracy**: 样本准确率

### DDR分级分类
- **accuracy**: 准确率
- **quadratic_kappa**: 二次加权Kappa
- **linear_kappa**: 线性加权Kappa
- **macro_F1**: 宏平均F1分数
- **micro_F1**: 微平均F1分数
- **confusion_matrix**: 混淆矩阵

## 🚨 TODO事项

### 必须完成
1. **实现数据加载器**
   - 创建ODIR数据集类
   - 创建DDR数据集类
   - 处理多标签和分级标签

2. **连接你的模型**
   - 在`YourModelInterface`中实现你的模型
   - 确保输入输出格式一致

3. **准备预训练权重**
   - 下载MAE预训练权重
   - 下载DINO预训练权重
   - 下载RETFound预训练权重

### 可选完成
1. **添加更多基线**
   - 实现更多CNN基线
   - 实现更多Transformer基线
   - 实现更多多任务学习方法

2. **优化训练策略**
   - 添加学习率调度器
   - 添加早停机制
   - 添加梯度裁剪

3. **增强可视化**
   - 添加更多图表类型
   - 添加交互式可视化
   - 添加3D可视化

## 📝 使用示例

### 完整流程示例

```python
# 1. 导入必要模块
from baselines.training.train_baseline import train_baseline
from baselines.evaluation.evaluate_baseline import evaluate_baseline, save_evaluation_results
from baselines.visualization.visualize import TopLevelVisualizer, load_results_from_json

# 2. 训练所有基线
baseline_models = ['resnet50', 'densenet121', 'vit_b', 'swin_t']

for model_name in baseline_models:
    train_baseline(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        task='odir'
    )

# 3. 评估所有基线
for model_name in baseline_models:
    metrics = evaluate_baseline(
        model_name=model_name,
        test_loader=test_loader,
        task='odir'
    )
    save_evaluation_results(model_name, 'odir', metrics)

# 4. 训练你的方法
# TODO: 实现你的方法训练
# train_your_method(...)

# 5. 评估你的方法
# TODO: 评估你的方法
# our_metrics = evaluate_your_method(...)

# 6. 生成可视化
visualizer = TopLevelVisualizer()
results = load_results_from_json(['results/*.json'])

visualizer.plot_performance_comparison(results, ['mAP', 'macro_F1'])
visualizer.plot_radar_chart(results, ['mAP', 'macro_F1', 'macro_AUC'])
visualizer.create_comparison_table(results, ['mAP', 'macro_F1'])
```

## 🎯 输出成果

### 训练输出
- 模型检查点 (`.pth`文件)
- 训练历史 (`.json`文件)
- 最佳模型指标

### 评估输出
- 评估指标 (`.json`文件)
- 混淆矩阵 (`.json`文件)

### 可视化输出
- 性能对比图 (`.png`文件)
- 消融实验图 (`.png`文件)
- 训练曲线图 (`.png`文件)
- 雷达图 (`.png`文件)
- 对比表格 (`.png`文件)

## 💡 最佳实践

1. **数据准备**
   - 确保数据集划分正确
   - 验证数据加载器工作正常
   - 检查标签格式

2. **训练监控**
   - 使用TensorBoard监控训练
   - 定期保存检查点
   - 记录训练日志

3. **评估验证**
   - 在多个数据集上评估
   - 计算统计显著性
   - 进行错误分析

4. **可视化优化**
   - 调整图表样式
   - 选择合适的图表类型
   - 确保图表清晰易读

## 📞 获取帮助

如果遇到问题：
1. 检查数据加载器是否正确
2. 检查模型配置是否正确
3. 检查设备是否可用
4. 查看训练日志和错误信息

---

**版本**: 1.0
**最后更新**: 2026-04-30
**维护者**: Research Team