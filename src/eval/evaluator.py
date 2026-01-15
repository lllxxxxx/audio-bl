"""
评估模块
"""
import torch
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from collections import defaultdict
import json
import os

from src.data.processor import parse_triplets, normalize_triplet


class Evaluator:
    """评估器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        processor,
        device: torch.device,
        max_new_tokens: int = 256
    ):
        """
        初始化评估器
        
        Args:
            model: 模型
            tokenizer: 分词器
            processor: 处理器
            device: 设备
            max_new_tokens: 最大生成token数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # 存储预测结果
        self.predictions = []
        
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        在数据集上评估模型
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        self.predictions = []
        
        all_pred_triplets = []
        all_gold_triplets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if batch is None:
                    continue
                
                # 生成预测
                predictions = self._generate_predictions(batch)
                
                # 解析真实标签和预测结果
                for i, pred_text in enumerate(predictions):
                    gold_text = batch['target_texts'][i]
                    sample_id = batch['ids'][i]
                    
                    # 解析三元组
                    pred_triplets = parse_triplets(pred_text)
                    gold_triplets = parse_triplets(gold_text)
                    
                    # 标准化
                    pred_triplets_norm = [normalize_triplet(t) for t in pred_triplets]
                    gold_triplets_norm = [normalize_triplet(t) for t in gold_triplets]
                    
                    all_pred_triplets.append(set(pred_triplets_norm))
                    all_gold_triplets.append(set(gold_triplets_norm))
                    
                    # 存储预测结果
                    self.predictions.append({
                        'id': sample_id,
                        'ground_truth': gold_text,
                        'prediction': pred_text,
                        'gold_triplets': [list(t) for t in gold_triplets],
                        'pred_triplets': [list(t) for t in pred_triplets],
                        'correct': pred_triplets_norm == gold_triplets_norm
                    })
        
        # 计算指标
        metrics = self._compute_metrics(all_pred_triplets, all_gold_triplets)
        
        return metrics
    
    def _generate_predictions(self, batch) -> List[str]:
        """生成预测"""
        # 准备输入
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # 添加音频特征（如果有）
        if 'input_features' in batch:
            model_inputs['input_features'] = batch['input_features'].to(self.device)
        if 'feature_attention_mask' in batch:
            model_inputs['feature_attention_mask'] = batch['feature_attention_mask'].to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 解码
        # 只取生成的部分
        generated_ids = generated_ids[:, input_ids.shape[1]:]
        predictions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return predictions
    
    def _compute_metrics(
        self,
        pred_triplets: List[set],
        gold_triplets: List[set]
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            pred_triplets: 预测的三元组集合列表
            gold_triplets: 真实的三元组集合列表
            
        Returns:
            包含precision, recall, micro_f1的字典
        """
        # 计算TP, FP, FN
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred_set, gold_set in zip(pred_triplets, gold_triplets):
            # True Positives: 预测正确的三元组
            tp = len(pred_set & gold_set)
            # False Positives: 预测了但不在真实标签中
            fp = len(pred_set - gold_set)
            # False Negatives: 真实标签中有但没预测出来
            fn = len(gold_set - pred_set)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # 计算Precision, Recall, Micro-F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'micro_f1': micro_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        }
    
    def get_sample_predictions(self, dataloader, num_samples: int = 5) -> List[Dict]:
        """
        获取样本预测结果用于日志记录
        
        Args:
            dataloader: 数据加载器
            num_samples: 样本数量
            
        Returns:
            预测样本列表
        """
        if self.predictions:
            return self.predictions[:num_samples]
        
        # 如果还没有预测结果，先运行评估
        self.evaluate(dataloader)
        return self.predictions[:num_samples]
    
    def save_predictions(self, save_path: str):
        """
        保存所有预测结果
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=2)
        
        print(f"Predictions saved to {save_path}")
    
    def print_metrics_report(self, metrics: Dict[str, float]):
        """打印指标报告"""
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Micro-F1:  {metrics['micro_f1']:.4f}")
        print("-"*50)
        print(f"True Positives:  {metrics['total_tp']}")
        print(f"False Positives: {metrics['total_fp']}")
        print(f"False Negatives: {metrics['total_fn']}")
        print("="*50)


def run_test_evaluation(
    model,
    tokenizer,
    processor,
    test_loader,
    device: torch.device,
    output_dir: str
) -> Dict[str, float]:
    """
    运行测试集评估
    
    Args:
        model: 模型
        tokenizer: 分词器
        processor: 处理器
        test_loader: 测试数据加载器
        device: 设备
        output_dir: 输出目录
        
    Returns:
        评估指标
    """
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device
    )
    
    # 评估
    metrics = evaluator.evaluate(test_loader)
    
    # 打印报告
    evaluator.print_metrics_report(metrics)
    
    # 保存预测结果
    predictions_path = os.path.join(output_dir, "test_predictions.json")
    evaluator.save_predictions(predictions_path)
    
    # 保存指标
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics
