"""
三维度增强模块

1. EntityAwareEnhancement - 专有名词识别增强
2. BoundaryContrastiveLoss - 实体边界约束
3. GroundingConstraintLoss - 幻觉抑制
"""

from .entity_aware import EntityAwareEnhancement
from .boundary_loss import BoundaryContrastiveLoss
from .grounding_loss import GroundingConstraintLoss

__all__ = [
    'EntityAwareEnhancement',
    'BoundaryContrastiveLoss', 
    'GroundingConstraintLoss'
]
