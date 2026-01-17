"""
训练模块 - 自定义训练循环
包含训练tricks: 权重衰减过滤、Cosine预热LR、Embedding噪声(NEFTune)
"""
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
import json

from src.data.template import get_template


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0
):
    """
    Cosine学习率调度器带预热
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: cosine周期数
        min_lr_ratio: 最小学习率比例
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        
        # 确保不低于最小学习率
        return max(min_lr_ratio, cosine_decay)
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def get_parameter_groups(
    model,
    learning_rate: float,
    weight_decay: float
) -> List[Dict[str, Any]]:
    """
    获取参数分组，对LayerNorm和bias不使用权重衰减
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        
    Returns:
        参数分组列表
    """
    # 不需要权重衰减的参数名称模式
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight", "ln_"]
    
    # 分组参数
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": learning_rate
        }
    ]
    
    # 打印分组统计
    decay_params = sum(p.numel() for p in optimizer_grouped_parameters[0]["params"])
    no_decay_params = sum(p.numel() for p in optimizer_grouped_parameters[1]["params"])
    print(f"Parameters with weight decay: {decay_params:,}")
    print(f"Parameters without weight decay (LayerNorm/bias): {no_decay_params:,}")
    
    return optimizer_grouped_parameters


class NEFTuneHook:
    """
    NEFTune: 在训练时给Embedding添加噪声
    论文: NEFTune: Noisy Embeddings Improve Instruction Finetuning
    """
    
    def __init__(self, model, neftune_noise_alpha: float = 5.0):
        """
        初始化NEFTune钩子
        
        Args:
            model: 模型
            neftune_noise_alpha: 噪声强度
        """
        self.model = model
        self.neftune_noise_alpha = neftune_noise_alpha
        self.handles = []
        self._original_forward = None
        
    def activate(self):
        """激活NEFTune"""
        # 找到embedding层
        embeddings = self._find_embedding_layer()
        if embeddings is None:
            print("Warning: Could not find embedding layer for NEFTune")
            return
        
        # 保存原始forward
        self._original_forward = embeddings.forward
        
        # 替换forward
        def neftune_forward(input_ids):
            # 原始embedding
            embeds = self._original_forward(input_ids)
            
            # 添加噪声
            if self.model.training:
                dims = torch.tensor(embeds.size(1) * embeds.size(2), dtype=torch.float32)
                mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)
                noise = torch.zeros_like(embeds).uniform_(-mag_norm.item(), mag_norm.item())
                embeds = embeds + noise
            
            return embeds
        
        embeddings.forward = neftune_forward
        print(f"NEFTune activated with alpha={self.neftune_noise_alpha}")
    
    def deactivate(self):
        """关闭NEFTune"""
        if self._original_forward is not None:
            embeddings = self._find_embedding_layer()
            if embeddings is not None:
                embeddings.forward = self._original_forward
            self._original_forward = None
    
    def _find_embedding_layer(self):
        """查找模型的embedding层"""
        # 常见的embedding层名称
        embedding_names = [
            "embed_tokens",
            "wte",
            "word_embeddings",
            "embedding",
            "embeddings.word_embeddings"
        ]
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                for embed_name in embedding_names:
                    if embed_name in name:
                        return module
        
        # 如果没找到，尝试获取第一个Embedding层
        for module in self.model.modules():
            if isinstance(module, nn.Embedding):
                return module
        
        return None


class Trainer:
    """
    自定义训练器
    
    特性:
    - 手写训练循环，方便添加创新点
    - LayerNorm和bias不使用权重衰减
    - Cosine学习率调度带预热
    - NEFTune embedding噪声
    - 梯度范数记录
    - 分数epoch记录
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        processor,
        config,
        train_loader,
        dev_loader,
        swanlab_run=None
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            tokenizer: 分词器
            processor: 处理器
            config: 训练配置
            train_loader: 训练数据加载器
            dev_loader: 验证数据加载器
            swanlab_run: SwanLab运行实例
        """
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.swanlab_run = swanlab_run
        
        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 参数分组 (LayerNorm和bias不使用weight decay)
        optimizer_grouped_parameters = get_parameter_groups(
            model=model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 优化器
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 计算总步数
        self.total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        # Cosine学习率调度器带预热
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
            min_lr_ratio=0.1  # 最小学习率为初始的10%
        )
        
        # NEFTune embedding噪声
        self.neftune_hook = None
        if getattr(config, 'use_neftune', True):
            neftune_alpha = getattr(config, 'neftune_noise_alpha', 5.0)
            self.neftune_hook = NEFTuneHook(model, neftune_noise_alpha=neftune_alpha)
            self.neftune_hook.activate()
        
        # 训练状态
        self.global_step = 0
        self.best_f1 = 0.0
        self.best_model_path = None
        
        # 每个epoch的step数
        self.steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        
        # 初始化三维度增强模块
        self.enhancements = self._init_enhancements()
        
        print(f"\nTrainer initialized:")
        print(f"  Total optimization steps: {self.total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  LR schedule: Cosine with warmup")
        print(f"  Weight decay: {config.weight_decay} (excluded for LayerNorm/bias)")
        print(f"  NEFTune: {'Enabled' if self.neftune_hook else 'Disabled'}")
        print(f"  Enhancements: {list(self.enhancements.keys()) if self.enhancements else 'None'}")
    
    def _init_enhancements(self):
        """初始化三维度增强模块，根据config决定是否启用"""
        enhancements = {}
        
        # 维度1: 专有名词识别增强
        if getattr(self.config, 'use_entity_aware', False):
            try:
                from src.enhancement.entity_aware import EntityAwareEnhancement
                # 获取模型的hidden_size
                hidden_size = getattr(self.model.config, 'hidden_size', 4096)
                enhancements['entity_aware'] = EntityAwareEnhancement(
                    hidden_size=hidden_size,
                    entity_boost=getattr(self.config, 'entity_boost_factor', 1.5)
                ).to(self.device)
                print(f"  Entity-Aware Enhancement: Enabled")
            except Exception as e:
                print(f"  Warning: Failed to init Entity-Aware Enhancement: {e}")
        
        # 维度2: 实体边界约束
        if getattr(self.config, 'use_boundary_loss', False):
            try:
                from src.enhancement.boundary_loss import BoundaryContrastiveLoss
                hidden_size = getattr(self.model.config, 'hidden_size', 4096)
                enhancements['boundary'] = BoundaryContrastiveLoss(
                    margin=getattr(self.config, 'boundary_margin', 1.0),
                    hidden_size=hidden_size
                )
                print(f"  Boundary Contrastive Loss: Enabled")
            except Exception as e:
                print(f"  Warning: Failed to init Boundary Loss: {e}")
        
        # 维度3: 幻觉抑制
        if getattr(self.config, 'use_grounding_loss', False):
            try:
                from src.enhancement.grounding_loss import GroundingConstraintLoss
                enhancements['grounding'] = GroundingConstraintLoss(
                    threshold=getattr(self.config, 'grounding_threshold', 0.1)
                )
                print(f"  Grounding Constraint Loss: Enabled")
            except Exception as e:
                print(f"  Warning: Failed to init Grounding Loss: {e}")
        
        return enhancements
    
    def _compute_enhancement_losses(self, batch, outputs):
        """计算增强损失（与SFT loss分离）"""
        enhancement_losses = {}
        
        # 维度1: Entity-Aware Loss
        if 'entity_aware' in self.enhancements:
            try:
                # 需要entity_mask从batch中获取
                entity_mask = batch.get('entity_mask')
                if entity_mask is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    entity_mask = entity_mask.to(self.device)
                    # 使用最后一层hidden states
                    hidden_states = outputs.hidden_states[-1]
                    _, entity_prob = self.enhancements['entity_aware'](hidden_states)
                    enhancement_losses['entity_aware'] = self.enhancements['entity_aware'].compute_loss(
                        entity_prob, entity_mask
                    )
            except Exception as e:
                pass  # 静默失败，不影响主训练
        
        # 维度2: Boundary Contrastive Loss
        if 'boundary' in self.enhancements:
            try:
                gold_triplets = batch.get('triplets', [])
                context_texts = batch.get('ground_truth_texts', [])
                if gold_triplets and context_texts and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # 使用最后一层hidden states的平均作为audio embedding
                    audio_embed = outputs.hidden_states[-1].mean(dim=1)
                    enhancement_losses['boundary'] = self.enhancements['boundary'].compute_loss(
                        audio_embed, gold_triplets, context_texts,
                        self.tokenizer, self.model, self.device
                    )
            except Exception as e:
                pass
        
        # 维度3: Grounding Constraint Loss
        if 'grounding' in self.enhancements:
            try:
                entity_positions = batch.get('entity_positions', [])
                if entity_positions and hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                    # 使用cross attention
                    cross_attn = outputs.cross_attentions[-1]  # 最后一层
                    enhancement_losses['grounding'] = self.enhancements['grounding'].compute_loss(
                        cross_attn, entity_positions
                    )
            except Exception as e:
                pass
        
        return enhancement_losses
        
    def train(self):
        """执行训练"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            
            # 每个epoch结束后在dev上验证
            print(f"\nValidating on dev set...")
            eval_results = self._evaluate(epoch)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Eval Precision: {eval_results['precision']:.4f}")
            print(f"  Eval Recall: {eval_results['recall']:.4f}")
            print(f"  Eval Micro-F1: {eval_results['micro_f1']:.4f}")
            
            # 记录到SwanLab
            if self.swanlab_run:
                self.swanlab_run.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "eval/precision": eval_results['precision'],
                    "eval/recall": eval_results['recall'],
                    "eval/micro_f1": eval_results['micro_f1']
                })
            
            # 保存最佳模型
            if eval_results['micro_f1'] > self.best_f1:
                self.best_f1 = eval_results['micro_f1']
                self._save_checkpoint(epoch, eval_results, is_best=True)
        
        # 关闭NEFTune
        if self.neftune_hook:
            self.neftune_hook.deactivate()
        
        print(f"\nTraining completed!")
        print(f"Best Micro-F1 on dev: {self.best_f1:.4f}")
        
        return self.best_model_path
    
    def _compute_grad_norm(self) -> float:
        """计算梯度范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            # 处理其他可能的输入
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            # 添加音频特征（如果有）
            if 'input_features' in batch:
                model_inputs['input_features'] = batch['input_features'].to(self.device)
            if 'feature_attention_mask' in batch:
                model_inputs['feature_attention_mask'] = batch['feature_attention_mask'].to(self.device)
            
            # 如果启用了增强模块，需要输出hidden_states和attentions
            if self.enhancements:
                model_inputs['output_hidden_states'] = True
                model_inputs['output_attentions'] = True
            
            # 前向传播 (bf16)
            if self.config.bf16 and torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(**model_inputs)
                    sft_loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(**model_inputs)
                sft_loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # 计算增强损失
            enhancement_losses = {}
            total_enhancement_loss = 0.0
            if self.enhancements:
                enhancement_losses = self._compute_enhancement_losses(batch, outputs)
                for name, enh_loss in enhancement_losses.items():
                    weight = getattr(self.config, f'{name}_weight', 0.1)
                    if name == 'entity_aware':
                        weight = getattr(self.config, 'entity_aware_weight', 0.1)
                    elif name == 'boundary':
                        weight = getattr(self.config, 'boundary_loss_weight', 0.1)
                    elif name == 'grounding':
                        weight = getattr(self.config, 'grounding_loss_weight', 0.1)
                    total_enhancement_loss += weight * enh_loss
            
            # 总损失 = SFT损失 + 增强损失
            loss = sft_loss + total_enhancement_loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # 梯度累积完成，进行优化步骤
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 计算梯度范数（在clip之前）
                grad_norm = self._compute_grad_norm() if self.config.log_grad_norm else 0.0
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 计算当前的fractional epoch
                current_epoch_progress = (step + 1) / len(self.train_loader)
                fractional_epoch = epoch + current_epoch_progress
                
                # 日志记录
                should_log = (
                    self.global_step % self.config.logging_steps == 0 or
                    (self.config.logging_first_step and self.global_step == 1)
                )
                
                if should_log:
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'grad_norm': f'{grad_norm:.2f}'
                    })
                    
                    if self.swanlab_run:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/global_step": self.global_step,
                            "train/epoch": round(fractional_epoch, 2),
                        }
                        
                        if self.config.log_grad_norm:
                            log_dict["train/grad_norm"] = grad_norm
                        
                        self.swanlab_run.log(log_dict)
                
                # 重置累积损失
                accumulated_loss = 0.0
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        """评估模型"""
        from src.eval.evaluator import Evaluator
        
        evaluator = Evaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            device=self.device
        )
        
        results = evaluator.evaluate(self.dev_loader)
        
        # 保存dev预测结果到文件
        dev_predictions_path = os.path.join(
            self.config.output_dir, 
            f"dev_predictions_epoch{epoch + 1}.json"
        )
        evaluator.save_predictions(dev_predictions_path)
        
        # 记录预测样本到SwanLab
        if self.swanlab_run and self.config.log_predictions:
            self._log_predictions(evaluator.get_sample_predictions(
                self.dev_loader, 
                num_samples=self.config.num_predictions_to_log
            ))
        
        return results
    
    def _log_predictions(self, predictions: list):
        """记录预测样本到SwanLab"""
        if not self.swanlab_run or not predictions:
            return
        
        # 将预测结果格式化为文本记录（swanlab.Table 在当前版本不可用）
        import swanlab
        
        # 构建文本格式的预测结果
        prediction_lines = []
        for pred in predictions:
            status = "✓" if pred['correct'] else "✗"
            prediction_lines.append(
                f"[{status}] ID: {pred['id']}\n"
                f"  GT: {pred['ground_truth']}\n"
                f"  Pred: {pred['prediction']}"
            )
        
        prediction_text = "\n\n".join(prediction_lines)
        
        # 使用 swanlab.Text 记录预测样本
        try:
            self.swanlab_run.log({
                "predictions": swanlab.Text(prediction_text)
            })
        except Exception as e:
            # 如果 swanlab.Text 也不可用，直接记录为字符串
            print(f"Warning: Could not log predictions to SwanLab: {e}")
    
    def _save_checkpoint(self, epoch: int, eval_results: dict, is_best: bool = False):
        """保存检查点"""
        from src.model.model import save_model
        
        if is_best:
            save_path = os.path.join(self.config.output_dir, "best_model")
        else:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        
        save_model(self.model, self.tokenizer, save_path)
        
        # 保存训练状态
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_f1': self.best_f1,
            'eval_results': eval_results,
            'config': {
                'learning_rate': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'warmup_ratio': self.config.warmup_ratio,
                'use_neftune': getattr(self.config, 'use_neftune', True),
                'neftune_noise_alpha': getattr(self.config, 'neftune_noise_alpha', 5.0)
            }
        }
        
        with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        if is_best:
            self.best_model_path = save_path
            print(f"New best model saved to {save_path}")
