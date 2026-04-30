"""
基线对比主脚本 - 一键运行所有基线对比实验

功能：
1. 训练所有基线模型
2. 评估所有基线模型
3. 生成对比可视化
4. 生成对比表格
"""

import argparse
import json
from pathlib import Path
import sys

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from baselines.training.train_baseline import train_baseline
from baselines.evaluation.evaluate_baseline import evaluate_baseline, save_evaluation_results
from baselines.visualization.visualize import TopLevelVisualizer, load_results_from_json


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基线对比实验主脚本')
    
    parser.add_argument('--mode', type=str, default='all',
                      choices=['train', 'evaluate', 'visualize', 'all'],
                      help='运行模式')
    
    parser.add_argument('--models', type=str, nargs='+',
                      default=['resnet50', 'densenet121', 'vit_b', 'swin_t'],
                      help='要训练/评估的模型列表')
    
    parser.add_argument('--task', type=str, default='odir',
                      choices=['odir', 'ddr', 'multi_task'],
                      help='任务类型')
    
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='训练轮数')
    
    parser.add_argument('--batch_size', type=int, default=16,
                      help='批次大小')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    
    parser.add_argument('--device', type=str, default='cuda',
                      help='设备')
    
    parser.add_argument('--data_dir', type=str, default='data/final_split',
                      help='数据目录')
    
    parser.add_argument('--output_dir', type=str, default='baselines/results',
                      help='输出目录')
    
    return parser.parse_args()


def run_training(args):
    """运行训练"""
    print("=" * 60)
    print("开始训练基线模型")
    print("=" * 60)
    
    # TODO: 导入你的数据加载器
    # from your_dataloader import ODIRDataset, DDRDataset
    # from torch.utils.data import DataLoader
    
    # TODO: 创建数据加载器
    # if args.task == 'odir':
    #     train_dataset = ODIRDataset(f'{args.data_dir}/ODIR/train')
    #     val_dataset = ODIRDataset(f'{args.data_dir}/ODIR/val')
    # elif args.task == 'ddr':
    #     train_dataset = DDRDataset(f'{args.data_dir}/DDR/train')
    #     val_dataset = DDRDataset(f'{args.data_dir}/DDR/val')
    # else:
    #     # 多任务数据集
    #     pass
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"任务类型: {args.task}")
    print(f"模型列表: {args.models}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"设备: {args.device}")
    
    # 训练每个模型
    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"训练 {model_name}")
        print(f"{'=' * 60}")
        
        try:
            best_acc = train_baseline(
                model_name=model_name,
                train_loader=None,  # TODO: 替换为实际的train_loader
                val_loader=None,    # TODO: 替换为实际的val_loader
                num_epochs=args.num_epochs,
                task=args.task,
                learning_rate=args.learning_rate,
                device=args.device,
                save_dir=args.output_dir
            )
            
            print(f"{model_name} 训练完成！最佳准确率: {best_acc:.4f}")
            
        except Exception as e:
            print(f"{model_name} 训练失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print("=" * 60)


def run_evaluation(args):
    """运行评估"""
    print("=" * 60)
    print("开始评估基线模型")
    print("=" * 60)
    
    # TODO: 导入你的数据加载器
    # from your_dataloader import ODIRDataset, DDRDataset
    # from torch.utils.data import DataLoader
    
    # TODO: 创建测试数据加载器
    # if args.task == 'odir':
    #     test_dataset = ODIRDataset(f'{args.data_dir}/ODIR/test')
    # elif args.task == 'ddr':
    #     test_dataset = DDRDataset(f'{args.data_dir}/DDR/test')
    # else:
    #     # 多任务数据集
    #     pass
    
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"任务类型: {args.task}")
    print(f"模型列表: {args.models}")
    print(f"设备: {args.device}")
    
    # 评估每个模型
    all_results = {}
    
    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"评估 {model_name}")
        print(f"{'=' * 60}")
        
        try:
            checkpoint_path = Path(args.output_dir) / f"{model_name}_{args.task}_best.pth"
            
            if not checkpoint_path.exists():
                print(f"检查点不存在: {checkpoint_path}")
                continue
            
            metrics = evaluate_baseline(
                model_name=model_name,
                test_loader=None,  # TODO: 替换为实际的test_loader
                task=args.task,
                checkpoint_path=str(checkpoint_path),
                device=args.device
            )
            
            all_results[model_name] = metrics
            save_evaluation_results(model_name, args.task, metrics, args.output_dir)
            
            print(f"{model_name} 评估完成！")
            print(f"主要指标:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
        except Exception as e:
            print(f"{model_name} 评估失败: {e}")
            continue
    
    # 保存所有结果
    all_results_path = Path(args.output_dir) / f"all_{args.task}_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("所有模型评估完成！")
    print(f"结果已保存到: {all_results_path}")
    print("=" * 60)
    
    return all_results


def run_visualization(args):
    """运行可视化"""
    print("=" * 60)
    print("开始生成可视化")
    print("=" * 60)
    
    # 加载所有结果
    all_results_path = Path(args.output_dir) / f"all_{args.task}_results.json"
    
    if not all_results_path.exists():
        print(f"结果文件不存在: {all_results_path}")
        print("请先运行评估模式")
        return
    
    with open(all_results_path, 'r') as f:
        all_results = json.load(f)
    
    print(f"加载了 {len(all_results)} 个模型的结果")
    
    # 创建可视化器
    visualizer = TopLevelVisualizer(output_dir=f'{args.output_dir}/figures')
    
    # 选择指标
    if args.task == 'odir':
        metrics = ['mAP', 'macro_F1', 'macro_AUC', 'macro_Precision', 'macro_Recall']
    elif args.task == 'ddr':
        metrics = ['accuracy', 'quadratic_kappa', 'macro_F1', 'macro_Precision', 'macro_Recall']
    else:
        metrics = ['ODIR_sample_accuracy', 'DDR_accuracy', 'ODIR_macro_F1', 'DDR_macro_F1']
    
    # 生成可视化
    print("\n生成性能对比图...")
    visualizer.plot_performance_comparison(
        results=all_results,
        metrics=metrics[:3],  # 前3个指标
        title=f'Performance Comparison ({args.task.upper()})',
        save_name=f'performance_comparison_{args.task}'
    )
    
    print("\n生成雷达图...")
    visualizer.plot_radar_chart(
        results=all_results,
        metrics=metrics[:3],
        title=f'Multi-Metric Comparison ({args.task.upper()})',
        save_name=f'radar_chart_{args.task}'
    )
    
    print("\n生成对比表格...")
    visualizer.create_comparison_table(
        results=all_results,
        metrics=metrics,
        save_name=f'comparison_table_{args.task}'
    )
    
    print("\n" + "=" * 60)
    print("所有可视化生成完成！")
    print(f"图表已保存到: {args.output_dir}/figures/")
    print("=" * 60)


def run_all(args):
    """运行完整流程"""
    print("=" * 60)
    print("开始完整基线对比流程")
    print("=" * 60)
    
    # 1. 训练
    print("\n阶段 1: 训练基线模型")
    print("-" * 60)
    run_training(args)
    
    # 2. 评估
    print("\n阶段 2: 评估基线模型")
    print("-" * 60)
    all_results = run_evaluation(args)
    
    # 3. 可视化
    print("\n阶段 3: 生成可视化")
    print("-" * 60)
    run_visualization(args)
    
    print("\n" + "=" * 60)
    print("完整基线对比流程完成！")
    print("=" * 60)
    
    # 打印总结
    print("\n总结:")
    print(f"  训练模型数: {len(args.models)}")
    print(f"  任务类型: {args.task}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  可视化图表: {args.output_dir}/figures/")


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("基线对比实验主脚本")
    print("=" * 60)
    print(f"模式: {args.mode}")
    print(f"任务: {args.task}")
    print(f"模型: {args.models}")
    print(f"设备: {args.device}")
    print("=" * 60)
    
    # 根据模式运行
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'evaluate':
        run_evaluation(args)
    elif args.mode == 'visualize':
        run_visualization(args)
    elif args.mode == 'all':
        run_all(args)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()