"""
维度2: Boundary Contrastive Learning

实体边界约束模块
核心思想：通过对比学习让模型区分正确边界和错误边界
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import random


class BoundaryContrastiveLoss(nn.Module):
    """
    实体边界约束
    
    原理：
    1. 从gold triplet构造正样本（正确边界）
    2. 构造边界错误的负样本（截断、过长、相邻词合并等）
    3. 使用Triplet Loss让模型区分正确和错误的边界
    """
    
    def __init__(self, margin: float = 1.0, hidden_size: int = 4096):
        """
        Args:
            margin: Triplet Loss的margin
            hidden_size: 嵌入维度
        """
        super().__init__()
        self.margin = margin
        self.hidden_size = hidden_size
        
        # 三元组编码器：将(subject, object, relation)编码为向量
        # 使用简单的线性层，实际训练时可以共享模型的text encoder
        self.triplet_encoder = None  # 将在forward时使用tokenizer和model
        
    def create_negative_samples(
        self, 
        gold_triplet: Tuple[str, str, str], 
        context_text: str,
        num_negatives: int = 3
    ) -> List[Tuple[str, str, str]]:
        """
        构造边界错误的负样本
        
        Args:
            gold_triplet: (subject, object, relation) 正确的三元组
            context_text: 原文
            num_negatives: 生成的负样本数量
            
        Returns:
            negatives: 边界错误的三元组列表
        """
        subj, obj, rel = gold_triplet
        negatives = []
        
        # 负样本类型1：实体截断
        if len(subj.split()) > 1:
            truncated = ' '.join(subj.split()[:-1])
            negatives.append((truncated, obj, rel))
        
        if len(obj.split()) > 1:
            truncated = ' '.join(obj.split()[:-1])
            negatives.append((subj, truncated, rel))
        
        # 负样本类型2：实体过长（吸收上下文）
        words = context_text.split()
        
        # 找subject在context中的位置
        try:
            for i, word in enumerate(words):
                if subj.split()[0].lower() in word.lower() and i > 0:
                    # 向前扩展
                    extended = words[i-1] + " " + subj
                    if extended != subj:
                        negatives.append((extended, obj, rel))
                        break
        except Exception:
            pass
        
        # 找object在context中的位置
        try:
            for i, word in enumerate(words):
                if obj.split()[0].lower() in word.lower() and i > 0:
                    # 向前扩展
                    extended = words[i-1] + " " + obj
                    if extended != obj:
                        negatives.append((subj, extended, rel))
                        break
        except Exception:
            pass
        
        # 负样本类型3：相邻词合并错误
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:!?')
            obj_first_word = obj.split()[0].lower() if obj else ""
            
            if word_lower == obj_first_word and i > 0:
                merged = words[i-1] + " " + word
                if merged.lower() != obj.lower():
                    negatives.append((subj, merged, rel))
                    break
        
        # 如果负样本不够，添加随机扰动
        while len(negatives) < num_negatives:
            # 随机截断或扩展
            if random.random() > 0.5 and len(subj) > 3:
                # 截断subject
                negatives.append((subj[:-2], obj, rel))
            elif len(obj) > 3:
                # 截断object
                negatives.append((subj, obj[:-2], rel))
            else:
                break
        
        return negatives[:num_negatives]
    
    def encode_triplet(
        self, 
        triplet: Tuple[str, str, str],
        tokenizer,
        model,
        device
    ) -> torch.Tensor:
        """
        将三元组编码为向量
        
        Args:
            triplet: (subject, object, relation)
            tokenizer: 分词器
            model: 用于编码的模型
            device: 设备
            
        Returns:
            embedding: [hidden_size] 三元组的嵌入向量
        """
        subj, obj, rel = triplet
        # 将三元组格式化为文本
        text = f"{subj} | {obj} | {rel}"
        
        # 编码
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 使用model的encoder部分获取embedding
            # 这里假设model有get_input_embeddings方法
            embeddings = model.get_input_embeddings()(inputs['input_ids'])
            # 取平均作为句子表示
            mask = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        
        return pooled.squeeze(0)
    
    def compute_loss(
        self,
        audio_embed: torch.Tensor,
        gold_triplets: List[Tuple[str, str, str]],
        context_texts: List[str],
        tokenizer,
        model,
        device
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            audio_embed: [batch, hidden_size] 音频嵌入
            gold_triplets: 正确的三元组列表
            context_texts: 原文列表
            tokenizer: 分词器
            model: 模型
            device: 设备
            
        Returns:
            loss: 对比损失
        """
        total_loss = 0.0
        valid_count = 0
        
        for i, (gold_triplet, context_text) in enumerate(zip(gold_triplets, context_texts)):
            # 编码正样本
            try:
                pos_embed = self.encode_triplet(gold_triplet, tokenizer, model, device)
            except Exception:
                continue
            
            # 生成并编码负样本
            negatives = self.create_negative_samples(gold_triplet, context_text)
            if not negatives:
                continue
            
            neg_embeds = []
            for neg in negatives:
                try:
                    neg_embed = self.encode_triplet(neg, tokenizer, model, device)
                    neg_embeds.append(neg_embed)
                except Exception:
                    continue
            
            if not neg_embeds:
                continue
            
            neg_embeds = torch.stack(neg_embeds, dim=0)  # [num_neg, hidden]
            
            # 使用音频嵌入作为anchor
            anchor = audio_embed[i] if audio_embed.dim() > 1 else audio_embed
            
            # 计算Triplet Loss
            # 对每个负样本计算loss，取平均
            for neg_embed in neg_embeds:
                loss = F.triplet_margin_loss(
                    anchor.unsqueeze(0),
                    pos_embed.unsqueeze(0),
                    neg_embed.unsqueeze(0),
                    margin=self.margin
                )
                total_loss += loss
                valid_count += 1
        
        if valid_count > 0:
            return total_loss / valid_count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


def create_boundary_contrastive_samples(gold_triplet, context_text):
    """
    便捷函数：构造边界相关的正负样本对
    
    Args:
        gold_triplet: (subject, object, relation)
        context_text: 原文
        
    Returns:
        positives: 正样本列表
        negatives: 负样本列表
    """
    loss_fn = BoundaryContrastiveLoss()
    positives = [gold_triplet]
    negatives = loss_fn.create_negative_samples(gold_triplet, context_text)
    return positives, negatives
