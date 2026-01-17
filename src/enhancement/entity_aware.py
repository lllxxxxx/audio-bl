"""
维度1: Entity-Aware Audio Encoding

专有名词识别增强模块
核心思想：让模型在编码音频时，对可能是专有名词的片段给予更多关注
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class EntityAwareEnhancement(nn.Module):
    """
    专有名词识别增强
    
    原理：
    1. 在音频hidden states上训练一个检测器，识别专有名词片段
    2. 对专有名词片段的表示进行增强
    3. 使用数据集自带的ground_truth文本和triplets构造监督信号
    """
    
    def __init__(self, hidden_size: int, entity_boost: float = 1.5):
        """
        Args:
            hidden_size: 模型hidden state的维度
            entity_boost: 专有名词片段的增强系数
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.entity_boost = entity_boost
        
        # 专有名词检测器：检测音频中哪些片段可能是专有名词
        self.entity_detector = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_size // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：检测专有名词片段并增强
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] 音频编码后的hidden states
            
        Returns:
            enhanced: [batch, seq_len, hidden_size] 增强后的hidden states
            entity_prob: [batch, seq_len, 1] 每个位置是专有名词的概率
        """
        # 检测专有名词片段
        # Conv1d expects [batch, channels, seq_len]
        entity_prob = self.entity_detector(hidden_states.transpose(1, 2))  # [batch, 1, seq_len]
        entity_prob = entity_prob.transpose(1, 2)  # [batch, seq_len, 1]
        
        # 对专有名词片段增强表示
        enhanced = hidden_states * (1 + (self.entity_boost - 1) * entity_prob)
        
        return enhanced, entity_prob
    
    def compute_loss(
        self, 
        entity_prob: torch.Tensor, 
        entity_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算实体检测损失
        
        Args:
            entity_prob: [batch, seq_len, 1] 模型预测的实体概率
            entity_mask: [batch, seq_len] 真实的实体位置（1=实体，0=非实体）
            attention_mask: [batch, seq_len] 有效位置mask
            
        Returns:
            loss: 实体检测的BCE损失
        """
        entity_prob = entity_prob.squeeze(-1)  # [batch, seq_len]
        
        if attention_mask is not None:
            # 只计算有效位置的损失
            loss = F.binary_cross_entropy(
                entity_prob * attention_mask.float(),
                entity_mask.float() * attention_mask.float(),
                reduction='sum'
            ) / (attention_mask.sum() + 1e-8)
        else:
            loss = F.binary_cross_entropy(entity_prob, entity_mask.float())
        
        return loss


def create_entity_mask(
    text: str, 
    triplets: List[Tuple[str, str, str]], 
    tokenizer,
    max_length: int = 512
) -> torch.Tensor:
    """
    从triplets提取实体，在text中定位，生成实体位置mask
    
    Args:
        text: 原始文本（数据集自带的ground_truth）
        triplets: 关系三元组列表 [(subject, object, relation), ...]
        tokenizer: 分词器
        max_length: 最大长度
        
    Returns:
        entity_mask: [seq_len] 1表示实体位置，0表示非实体
    """
    # Tokenize文本
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
    entity_mask = torch.zeros(len(tokens))
    
    # 提取所有实体
    entities = []
    for triplet in triplets:
        if len(triplet) >= 3:
            entities.append(triplet[0])  # subject
            entities.append(triplet[1])  # object
    
    # 对于每个实体，找到它在token序列中的位置
    text_lower = text.lower()
    for entity in entities:
        entity_lower = entity.lower()
        
        # 在原文中找到实体位置
        start_char = text_lower.find(entity_lower)
        if start_char == -1:
            continue
        end_char = start_char + len(entity)
        
        # 将字符位置转换为token位置
        # 使用tokenizer的offset_mapping（如果支持）
        try:
            encoded = tokenizer(text, return_offsets_mapping=True, max_length=max_length, truncation=True)
            offset_mapping = encoded.get('offset_mapping', None)
            
            if offset_mapping is not None:
                for i, (start, end) in enumerate(offset_mapping):
                    if start is not None and end is not None:
                        # 如果token与实体有重叠
                        if start < end_char and end > start_char:
                            if i < len(entity_mask):
                                entity_mask[i] = 1
        except Exception:
            # 如果tokenizer不支持offset_mapping，使用简单的近似方法
            entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
            # 滑动窗口查找
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                    entity_mask[i:i+len(entity_tokens)] = 1
                    break
    
    return entity_mask
