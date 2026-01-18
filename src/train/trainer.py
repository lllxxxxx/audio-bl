"""
训练模块 - 自定义训练循环
包含训练tricks: 权重衰减过滤、Cosine预热LR、Embedding噪声(NEFTune)
支持解耦鲁棒性框架 (Decoupled Robustness Framework):
- BCL: 边界感知对比学习 (Boundary-aware Contrastive Learning)
- RDH: 反思性去幻觉 (Reflective De-Hallucination)
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    - 解耦鲁棒性框架 (BCL + RDH)
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
        
        # 解耦鲁棒性框架配置
        self.use_bcl = getattr(config, 'use_bcl', False)
        self.use_rdh = getattr(config, 'use_rdh', False)
        self.lambda_bcl = getattr(config, 'lambda_bcl', 0.1)
        self.lambda_rdh = getattr(config, 'lambda_rdh', 0.1)
        self.bcl_margin = getattr(config, 'bcl_margin', 0.5)
        
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
        
        print(f"\nTrainer initialized:")
        print(f"  Total optimization steps: {self.total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Steps per epoch: {self.steps_per_epoch}")
        print(f"  LR schedule: Cosine with warmup")
        print(f"  Weight decay: {config.weight_decay} (excluded for LayerNorm/bias)")
        print(f"  NEFTune: {'Enabled' if self.neftune_hook else 'Disabled'}")
        print(f"  BCL (Boundary-aware Contrastive Learning): {'Enabled' if self.use_bcl else 'Disabled'}")
        print(f"  RDH (Reflective De-Hallucination): {'Enabled' if self.use_rdh else 'Disabled'}")
        if self.use_bcl:
            print(f"    BCL lambda: {self.lambda_bcl}, margin: {self.bcl_margin}")
        if self.use_rdh:
            print(f"    RDH lambda: {self.lambda_rdh}")
    

        
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
            train_metrics = self._train_epoch(epoch)
            
            # 每个epoch结束后在dev上验证
            print(f"\nValidating on dev set...")
            eval_results = self._evaluate(epoch)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss (Total): {train_metrics['total_loss']:.4f}")
            if self.use_bcl:
                print(f"  Train Loss (SFT): {train_metrics['sft_loss']:.4f}")
                print(f"  Train Loss (BCL): {train_metrics['bcl_loss']:.4f}")
            if self.use_rdh:
                print(f"  Train Loss (RDH): {train_metrics['rdh_loss']:.4f}")
            print(f"  Eval Precision: {eval_results['precision']:.4f}")
            print(f"  Eval Recall: {eval_results['recall']:.4f}")
            print(f"  Eval Micro-F1: {eval_results['micro_f1']:.4f}")
            
            # 记录到SwanLab
            if self.swanlab_run:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_metrics['total_loss'],
                    "eval/precision": eval_results['precision'],
                    "eval/recall": eval_results['recall'],
                    "eval/micro_f1": eval_results['micro_f1']
                }
                if self.use_bcl:
                    log_dict["train/epoch_sft_loss"] = train_metrics['sft_loss']
                    log_dict["train/epoch_bcl_loss"] = train_metrics['bcl_loss']
                if self.use_rdh:
                    log_dict["train/epoch_rdh_loss"] = train_metrics['rdh_loss']
                self.swanlab_run.log(log_dict)
            
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
    

    
    def _extract_bcl_hidden_states(
        self,
        pos_entities: List[str],
        neg_entities: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取 BCL 正负样本的 hidden states
        
        使用 Teacher Forcing 模式：将正确/错误边界文本作为输入，
        提取解码器的 hidden state 用于对比学习。
        
        关键约束：不计算 CE Loss
        注意：仅使用纯文本 encoding，不传入 audio features
        
        Args:
            pos_entities: 正确边界实体列表
            neg_entities: 错误边界实体列表  
            
        Returns:
            (h_pos, h_neg): 正负样本的 hidden states
        """
        batch_size = len(pos_entities)
        
        # 编码正样本
        pos_encoded = self.tokenizer(
            pos_entities,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        pos_ids = pos_encoded.input_ids.to(self.device)
        pos_mask = pos_encoded.attention_mask.to(self.device)
        
        # 编码负样本
        neg_encoded = self.tokenizer(
            neg_entities,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        neg_ids = neg_encoded.input_ids.to(self.device)
        neg_mask = neg_encoded.attention_mask.to(self.device)
        
        # 正样本 forward (需要梯度)
        pos_inputs = {
            'input_ids': pos_ids,
            'attention_mask': pos_mask,
            'output_hidden_states': True,
        }
            
        outputs_pos = self.model(**pos_inputs)
        
        # 负样本 forward (需要梯度)
        neg_inputs = {
            'input_ids': neg_ids,
            'attention_mask': neg_mask,
            'output_hidden_states': True,
        }
            
        outputs_neg = self.model(**neg_inputs)
        
        # 提取最后一个 token 的 hidden state
        pos_hidden = outputs_pos.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        neg_hidden = outputs_neg.hidden_states[-1]
        
        pos_lengths = pos_mask.sum(dim=1) - 1
        neg_lengths = neg_mask.sum(dim=1) - 1
        
        h_pos = pos_hidden[
            torch.arange(batch_size, device=pos_ids.device),
            pos_lengths.long().clamp(min=0)
        ]
        h_neg = neg_hidden[
            torch.arange(batch_size, device=neg_ids.device),
            neg_lengths.long().clamp(min=0)
        ]
        
        return h_pos, h_neg
    
    def _get_audio_anchor_representation(
        self,
        outputs,
        input_features: torch.Tensor,
        feature_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取音频的 anchor 表示
        
        从模型输出中提取音频相关的表示，用于 BCL 对比学习
        
        Args:
            outputs: 模型输出
            input_features: 音频特征
            feature_attention_mask: 音频特征注意力掩码
            
        Returns:
            音频 anchor 表示 [batch_size, hidden_dim]
        """
        # 使用最后一层 hidden states 的音频部分
        # 由于 Qwen2-Audio 的特殊架构，音频特征会被编码到 hidden states 中
        # 这里我们使用一个简化的方法：取 hidden states 的平均作为 anchor
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            # 对序列维度取平均作为 anchor
            audio_anchor = last_hidden_states.mean(dim=1)  # [batch, hidden_dim]
        else:
            # 如果没有 hidden_states，使用 input_features 的平均
            # 这是一个 fallback 方案
            if input_features.dim() == 3:
                audio_anchor = input_features.mean(dim=1)  # [batch, hidden_dim]
            else:
                audio_anchor = input_features.mean(dim=-1)
        
        return audio_anchor
    
    def _compute_bcl_loss(
        self,
        audio_anchor: torch.Tensor,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
        margin: float = 0.5
    ) -> torch.Tensor:
        """
        计算边界感知对比学习 (BCL) 的 Margin Ranking Loss
        
        Loss = max(0, margin - sim(anchor, pos) + sim(anchor, neg))
        
        目标：让正样本比负样本更接近音频 anchor
        
        Args:
            audio_anchor: 音频表示 [batch, hidden_dim]
            h_pos: 正样本 (正确边界) hidden state [batch, hidden_dim]
            h_neg: 负样本 (错误边界) hidden state [batch, hidden_dim]
            margin: margin 值
            
        Returns:
            BCL loss (标量)
        """
        # 计算余弦相似度
        sim_pos = F.cosine_similarity(audio_anchor, h_pos, dim=-1)  # [batch]
        sim_neg = F.cosine_similarity(audio_anchor, h_neg, dim=-1)  # [batch]
        
        # Margin Ranking Loss
        # 我们希望 sim_pos > sim_neg，即 sim_pos - sim_neg > margin
        # 使用 margin_ranking_loss: loss = max(0, -y * (x1 - x2) + margin)
        # 当 y=1 时，loss = max(0, -(x1 - x2) + margin) = max(0, margin - x1 + x2)
        # 这里 x1 = sim_pos, x2 = sim_neg
        target = torch.ones_like(sim_pos)  # y = 1
        loss = F.margin_ranking_loss(sim_pos, sim_neg, target, margin=margin)
        
        return loss
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        实现三流训练:
        - 流 1: 标准 SFT (CE Loss)
        - 流 2: BCL 边界对比学习 (Margin Ranking Loss, 不计算 CE Loss)
        - 流 3: RDH 反思性去幻觉 (CE Loss)
        """
        self.model.train()
        
        # 分别累积各项 loss
        total_loss_sum = 0.0
        sft_loss_sum = 0.0
        bcl_loss_sum = 0.0
        rdh_loss_sum = 0.0
        num_batches = 0
        accumulated_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
            
            # ============== 流 1: 标准 SFT ==============
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'output_hidden_states': self.use_bcl  # BCL 需要 hidden states
            }
            
            # 添加音频特征
            input_features = None
            feature_attention_mask = None
            if 'input_features' in batch:
                input_features = batch['input_features'].to(self.device)
                model_inputs['input_features'] = input_features
            if 'feature_attention_mask' in batch:
                feature_attention_mask = batch['feature_attention_mask'].to(self.device)
                model_inputs['feature_attention_mask'] = feature_attention_mask
            
            # SFT 前向传播
            if self.config.bf16 and torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs_sft = self.model(**model_inputs)
                    loss_sft = outputs_sft.loss
            else:
                outputs_sft = self.model(**model_inputs)
                loss_sft = outputs_sft.loss
            
            total_loss = loss_sft
            current_sft_loss = loss_sft.item()
            current_bcl_loss = 0.0
            current_rdh_loss = 0.0
            
            # ============== 流 2: BCL (边界对比学习) ==============
            # BCL 目标: 让正确边界的特征比错误边界更接近音频 anchor
            # 关键: 不计算 CE Loss，只提取特征用于 Margin Ranking Loss
            if self.use_bcl and 'bcl_pos_entities' in batch and 'bcl_neg_entities' in batch:
                try:
                    bcl_pos_entities = batch['bcl_pos_entities']
                    bcl_neg_entities = batch['bcl_neg_entities']
                    
                    # 提取正负样本的 hidden states (Teacher Forcing, 无 CE Loss)
                    # 梯度会通过 h_pos 和 h_neg 反向传播更新模型
                    # 注意: BCL 实体是纯文本，不需要 input_features (避免 shape 错误)
                    h_pos, h_neg = self._extract_bcl_hidden_states(
                        bcl_pos_entities, 
                        bcl_neg_entities
                    )
                    
                    # 获取音频 anchor 表示 (从 SFT forward 中获取，detach 避免重复梯度)
                    audio_anchor = self._get_audio_anchor_representation(
                        outputs_sft, input_features, feature_attention_mask
                    ).detach()  # anchor 不参与梯度更新，只作为固定参照
                    
                    # 计算 BCL Margin Ranking Loss
                    # 目标: sim(anchor, h_pos) > sim(anchor, h_neg) + margin
                    loss_bcl = self._compute_bcl_loss(
                        audio_anchor,
                        h_pos,  # 有梯度
                        h_neg,  # 有梯度
                        margin=self.bcl_margin
                    )
                    
                    total_loss = total_loss + self.lambda_bcl * loss_bcl
                    current_bcl_loss = loss_bcl.item()
                    
                except Exception as e:
                    # BCL 失败时跳过
                    if step == 0:
                        print(f"Warning: BCL computation failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # ============== 流 3: RDH (反思性去幻觉) ==============
            if self.use_rdh and batch.get('rdh_input_ids') is not None:
                try:
                    rdh_input_ids = batch['rdh_input_ids'].to(self.device)
                    rdh_attention_mask = batch['rdh_attention_mask'].to(self.device)
                    rdh_labels = batch['rdh_labels'].to(self.device)
                    
                    rdh_model_inputs = {
                        'input_ids': rdh_input_ids,
                        'attention_mask': rdh_attention_mask,
                        'labels': rdh_labels  # 正确实体作为 label (CE Loss)
                    }
                    
                    # 添加 RDH 音频特征
                    if 'rdh_input_features' in batch:
                        rdh_model_inputs['input_features'] = batch['rdh_input_features'].to(self.device)
                    if 'rdh_feature_attention_mask' in batch:
                        rdh_model_inputs['feature_attention_mask'] = batch['rdh_feature_attention_mask'].to(self.device)
                    
                    # RDH 前向传播 (正常 CE Loss)
                    if self.config.bf16 and torch.cuda.is_available():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs_rdh = self.model(**rdh_model_inputs)
                            loss_rdh = outputs_rdh.loss
                    else:
                        outputs_rdh = self.model(**rdh_model_inputs)
                        loss_rdh = outputs_rdh.loss
                    
                    total_loss = total_loss + self.lambda_rdh * loss_rdh
                    current_rdh_loss = loss_rdh.item()
                    
                except Exception as e:
                    # RDH 失败时跳过
                    if step == 0:
                        print(f"Warning: RDH computation failed: {e}")
            
            # ============== 反向传播 ==============
            scaled_loss = total_loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += total_loss.item()
            total_loss_sum += total_loss.item()
            sft_loss_sum += current_sft_loss
            bcl_loss_sum += current_bcl_loss
            rdh_loss_sum += current_rdh_loss
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
                    
                    postfix_dict = {
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                    }
                    if self.config.log_grad_norm:
                        postfix_dict['grad_norm'] = f'{grad_norm:.2f}'
                    
                    progress_bar.set_postfix(postfix_dict)
                    
                    if self.swanlab_run:
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/loss_sft": current_sft_loss,
                            "train/learning_rate": lr,
                            "train/global_step": self.global_step,
                            "train/epoch": round(fractional_epoch, 2),
                        }
                        
                        if self.use_bcl:
                            log_dict["train/loss_bcl"] = current_bcl_loss
                        if self.use_rdh:
                            log_dict["train/loss_rdh"] = current_rdh_loss
                        if self.config.log_grad_norm:
                            log_dict["train/grad_norm"] = grad_norm
                        
                        self.swanlab_run.log(log_dict)
                
                # 重置累积损失
                accumulated_loss = 0.0
        
        # 返回各项 loss 的平均值
        num_batches = max(num_batches, 1)
        return {
            'total_loss': total_loss_sum / num_batches,
            'sft_loss': sft_loss_sum / num_batches,
            'bcl_loss': bcl_loss_sum / num_batches if self.use_bcl else 0.0,
            'rdh_loss': rdh_loss_sum / num_batches if self.use_rdh else 0.0
        }
    
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
                'neftune_noise_alpha': getattr(self.config, 'neftune_noise_alpha', 5.0),
                'use_bcl': self.use_bcl,
                'lambda_bcl': self.lambda_bcl,
                'bcl_margin': self.bcl_margin,
                'use_rdh': self.use_rdh,
                'lambda_rdh': self.lambda_rdh
            }
        }
        
        with open(os.path.join(save_path, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        if is_best:
            self.best_model_path = save_path
            print(f"New best model saved to {save_path}")
