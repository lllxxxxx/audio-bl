"""
维度3: Grounding Constraint Loss

幻觉抑制模块
核心思想：限制模型只能输出"有根据"的内容，必须能对应到音频片段
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class GroundingConstraintLoss(nn.Module):
    """
    幻觉抑制
    
    原理：
    1. 检查生成的实体对音频的cross-attention权重
    2. 如果实体对任何音频片段的注意力都很低 → 可能是幻觉
    3. 对这种情况给予惩罚
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: 最低attention阈值，低于此值认为是幻觉
        """
        super().__init__()
        self.threshold = threshold
        
    def forward(
        self, 
        cross_attention_weights: torch.Tensor, 
        entity_positions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        计算幻觉惩罚损失
        
        Args:
            cross_attention_weights: [batch, num_heads, gen_len, audio_len] 
                生成token对音频的交叉注意力权重
                或 [batch, gen_len, audio_len] 如果已经平均过heads
            entity_positions: List of (start, end) 生成实体的token位置
            
        Returns:
            loss: 幻觉惩罚损失
        """
        if len(entity_positions) == 0:
            return torch.tensor(0.0, device=cross_attention_weights.device, requires_grad=True)
        
        # 如果有多个head，先平均
        if cross_attention_weights.dim() == 4:
            # [batch, num_heads, gen_len, audio_len] -> [batch, gen_len, audio_len]
            cross_attention_weights = cross_attention_weights.mean(dim=1)
        
        total_loss = 0.0
        valid_count = 0
        
        for entity_start, entity_end in entity_positions:
            if entity_start >= cross_attention_weights.size(1) or entity_end > cross_attention_weights.size(1):
                continue
            
            # 实体token对音频的平均注意力
            # [batch, entity_len, audio_len] -> [batch, audio_len]
            entity_attn = cross_attention_weights[:, entity_start:entity_end, :].mean(dim=1)
            
            # 最大注意力值 [batch]
            max_attn = entity_attn.max(dim=-1)[0]
            
            # 如果实体对任何音频片段的注意力都很低 → 可能是幻觉
            # 惩罚: max(0, threshold - max_attn)
            hallucination_penalty = F.relu(self.threshold - max_attn)
            total_loss += hallucination_penalty.mean()
            valid_count += 1
        
        if valid_count > 0:
            return total_loss / valid_count
        else:
            return torch.tensor(0.0, device=cross_attention_weights.device, requires_grad=True)
    
    def compute_loss(
        self,
        cross_attention_weights: torch.Tensor,
        entity_positions: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        compute_loss的别名，保持接口一致
        """
        return self.forward(cross_attention_weights, entity_positions)


class SemanticVerifier:
    """
    语义规则验证器（可选辅助模块）
    
    基于规则检查triplet是否符合语义约束
    用于快速过滤明显错误的预测
    """
    
    # 关系类型约束规则
    RELATION_CONSTRAINTS = {
        "Live_In": {
            "subject_should_contain": ["person", "people", "he", "she", "they"],
            "object_should_contain": ["city", "country", "state", "place"],
        },
        "Work_For": {
            "subject_should_contain": ["person", "he", "she"],
            "object_should_contain": ["company", "organization", "org", "corp"],
        },
        "Located_In": {
            "disallow_same_entity": True,  # subject和object不能是同一个
        },
        "OrgBased_In": {
            "disallow_same_entity": True,
        },
        "Kill": {
            "subject_should_contain": ["person", "he", "she", "they"],
            "object_should_contain": ["person", "he", "she", "they"],
        }
    }
    
    def is_valid(self, triplet: Tuple[str, str, str]) -> bool:
        """
        检查triplet是否符合基本语义约束
        
        Args:
            triplet: (subject, object, relation)
            
        Returns:
            valid: 是否有效
        """
        subj, obj, rel = triplet
        
        # 规则1：Subject和Object不能相同或包含关系
        subj_lower = subj.lower().strip()
        obj_lower = obj.lower().strip()
        
        if subj_lower == obj_lower:
            return False
        
        if subj_lower in obj_lower or obj_lower in subj_lower:
            # 避免 "America" 和 "American" 这种情况
            return False
        
        # 规则2：检查关系特定约束
        constraints = self.RELATION_CONSTRAINTS.get(rel, {})
        if constraints.get("disallow_same_entity", False):
            if subj_lower == obj_lower:
                return False
        
        return True
    
    def compute_penalty(
        self, 
        pred_triplets: List[Tuple[str, str, str]], 
        gold_triplets: List[Tuple[str, str, str]]
    ) -> float:
        """
        计算语义违规惩罚
        
        Args:
            pred_triplets: 预测的三元组列表
            gold_triplets: 正确的三元组列表
            
        Returns:
            penalty: 惩罚值（违规数量/总数）
        """
        if not pred_triplets:
            return 0.0
        
        violations = 0
        for triplet in pred_triplets:
            if not self.is_valid(triplet):
                violations += 1
        
        return violations / len(pred_triplets)


def extract_entity_positions_from_output(
    output_ids: torch.Tensor,
    tokenizer,
    triplets: List[Tuple[str, str, str]]
) -> List[Tuple[int, int]]:
    """
    从生成输出中提取实体的token位置
    
    Args:
        output_ids: [seq_len] 生成的token ids
        tokenizer: 分词器
        triplets: 预测的三元组（包含要查找的实体）
        
    Returns:
        positions: List of (start, end) 实体在输出中的位置
    """
    positions = []
    
    # 解码输出
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # 提取所有实体
    entities = []
    for triplet in triplets:
        if len(triplet) >= 2:
            entities.append(triplet[0])  # subject
            entities.append(triplet[1])  # object
    
    # 对每个实体找到其token位置
    for entity in entities:
        try:
            # 找到实体在输出文本中的位置
            entity_lower = entity.lower()
            output_lower = output_text.lower()
            
            start_char = output_lower.find(entity_lower)
            if start_char == -1:
                continue
            
            # 编码找到token位置
            # 这是一个简化的方法，实际可能需要更精确的对齐
            prefix = output_text[:start_char]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
            
            start_token = len(prefix_tokens)
            end_token = start_token + len(entity_tokens)
            
            positions.append((start_token, end_token))
            
        except Exception:
            continue
    
    return positions
