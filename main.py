"""
Qwen2-Audio 语音关系抽取微调
主入口文件
"""
import argparse
import os
import random
import numpy as np
import torch
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config import TrainingConfig, InferenceConfig
from src.model.model import load_model_and_processor, load_trained_model
from src.data.dataset import create_dataloaders
from src.data.processor import AudioRECollator
from src.data.template import get_template
from src.train.trainer import Trainer
from src.eval.evaluator import run_test_evaluation


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_swanlab(config: TrainingConfig):
    """初始化SwanLab"""
    if not config.use_swanlab:
        return None
    
    try:
        import swanlab
        
        run = swanlab.init(
            project=config.swanlab_project,
            experiment_name=config.swanlab_experiment,
            config={
                "model": config.model_name_or_path,
                "dataset": config.dataset_name,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "bf16": config.bf16
            }
        )
        print(f"SwanLab initialized: {config.swanlab_project}/{config.swanlab_experiment}")
        return run
    except Exception as e:
        print(f"Failed to initialize SwanLab: {e}")
        return None


def train(args):
    """
    训练流程:
    1. 在train集上训练
    2. 每个epoch在dev集上验证
    3. 保存dev上最佳的模型
    4. 最后在test集上测试
    """
    print("="*60)
    print("Starting Training Pipeline")
    print("="*60)
    
    # 加载配置
    config = TrainingConfig(
        model_name_or_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=not args.no_lora,
        use_swanlab=not args.no_swanlab
    )
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 初始化SwanLab
    swanlab_run = init_swanlab(config)
    
    # 获取数据集模板
    template = get_template(config.dataset_name)
    print(f"Using template: {template.name}")
    print(f"Relation types: {template.relation_types}")
    
    # 加载模型
    lora_config = {
        'r': config.lora_rank,
        'lora_alpha': config.lora_alpha,
        'lora_dropout': config.lora_dropout,
        'target_modules': config.lora_target_modules
    }
    
    model, processor, tokenizer = load_model_and_processor(
        model_name_or_path=config.model_name_or_path,
        device=config.device,
        use_lora=config.use_lora,
        lora_config=lora_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    )
    
    # 创建数据整理器
    train_collator = AudioRECollator(
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=config.max_length,
        is_training=True
    )
    
    eval_collator = AudioRECollator(
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=config.max_length,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader, dev_loader, test_loader = create_dataloaders(
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        train_collator=train_collator,
        eval_collator=eval_collator
    )
    
    print(f"\nDataset: {config.dataset_name}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Dev samples: {len(dev_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        swanlab_run=swanlab_run
    )
    
    # 开始训练 (每个epoch在dev上验证，保存最佳模型)
    best_model_path = trainer.train()
    
    # 加载最佳模型在test集上测试
    print("\n" + "="*60)
    print("Testing Best Model on Test Set")
    print("="*60)
    
    # 保存最佳F1分数，稍后用于打印
    best_dev_f1 = trainer.best_f1
    
    # 释放训练模型占用的显存
    print("Releasing training model from GPU memory...")
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()
    print("GPU memory released.")
    
    # 重新加载最佳模型
    print(f"Loading best model from: {best_model_path}")
    best_model, _, _ = load_trained_model(
        base_model_path=config.model_name_or_path,
        lora_path=best_model_path,
        device=config.device,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32
    )
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    test_metrics = run_test_evaluation(
        model=best_model,
        tokenizer=tokenizer,
        processor=processor,
        test_loader=test_loader,
        device=device,
        output_dir=config.output_dir
    )
    
    # 记录最终测试指标
    if swanlab_run:
        swanlab_run.log({
            "test/precision": test_metrics['precision'],
            "test/recall": test_metrics['recall'],
            "test/micro_f1": test_metrics['micro_f1']
        })
        swanlab_run.finish()
    
    print("\n" + "="*60)
    print("Training and Evaluation Completed!")
    print("="*60)
    print(f"Best model saved at: {best_model_path}")
    print(f"Best dev Micro-F1: {best_dev_f1:.4f}")
    print(f"Test Micro-F1: {test_metrics['micro_f1']:.4f}")


def evaluate(args):
    """评估流程"""
    print("="*60)
    print("Starting Evaluation Pipeline")
    print("="*60)
    
    # 加载配置
    infer_config = InferenceConfig(
        model_path=args.model_path,
        device=args.device,
        dataset_name=args.dataset
    )
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset
    )
    
    # 获取模板
    template = get_template(config.dataset_name)
    
    # 加载训练好的模型
    model, processor, tokenizer = load_trained_model(
        base_model_path=args.base_model,
        lora_path=args.model_path,
        device=infer_config.device
    )
    
    # 创建评估数据整理器
    eval_collator = AudioRECollator(
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=512,
        is_training=False
    )
    
    # 创建数据加载器
    _, _, test_loader = create_dataloaders(
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        train_collator=eval_collator,  # 不使用
        eval_collator=eval_collator
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 评估
    device = torch.device(infer_config.device if torch.cuda.is_available() else "cpu")
    metrics = run_test_evaluation(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        test_loader=test_loader,
        device=device,
        output_dir=config.output_dir
    )
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen2-Audio Speech Relation Extraction")
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or eval')
    
    # 训练参数
    train_parser = subparsers.add_parser('train', help='Training mode')
    train_parser.add_argument('--model-path', type=str, 
                              default='Qwen/Qwen2-Audio-7B-Instruct',
                              help='Model path or name')
    train_parser.add_argument('--data-dir', type=str, default='./conll04',
                              help='Data directory')
    train_parser.add_argument('--output-dir', type=str, default='./output',
                              help='Output directory')
    train_parser.add_argument('--dataset', type=str, default='conll04',
                              help='Dataset name (conll04, nyt, webnlg, or custom)')
    train_parser.add_argument('--epochs', type=int, default=10,
                              help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=1,
                              help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                              help='Learning rate')
    train_parser.add_argument('--no-lora', action='store_true',
                              help='Disable LoRA')
    train_parser.add_argument('--no-swanlab', action='store_true',
                              help='Disable SwanLab logging')
    
    # 评估参数
    eval_parser = subparsers.add_parser('eval', help='Evaluation mode')
    eval_parser.add_argument('--model-path', type=str, required=True,
                             help='Path to trained LoRA model')
    eval_parser.add_argument('--base-model', type=str,
                             default='Qwen/Qwen2-Audio-7B-Instruct',
                             help='Base model path')
    eval_parser.add_argument('--data-dir', type=str, default='./conll04',
                             help='Data directory')
    eval_parser.add_argument('--output-dir', type=str, default='./output',
                             help='Output directory')
    eval_parser.add_argument('--dataset', type=str, default='conll04',
                             help='Dataset name')
    eval_parser.add_argument('--device', type=str, default='cuda',
                             help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
