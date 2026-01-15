"""
数据处理模块
"""
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from src.data.template import get_template, DatasetTemplate


class AudioRECollator:
    """音频关系抽取数据整理器"""
    
    def __init__(
        self,
        processor,
        tokenizer,
        template: DatasetTemplate,
        max_length: int = 512,
        is_training: bool = True
    ):
        """
        初始化整理器
        
        Args:
            processor: Qwen2-Audio处理器
            tokenizer: 分词器
            template: 数据集模板
            max_length: 最大序列长度
            is_training: 是否为训练模式
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.is_training = is_training
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 批次数据列表
            
        Returns:
            整理后的批次数据
        """
        ids = [item['id'] for item in batch]
        audios = [item['audio'].numpy() for item in batch]
        target_texts = [item['target_text'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]
        
        # 获取模板内容
        system_prompt = self.template.get_full_system_prompt()
        user_prompt = self.template.user_prompt
        
        # 构建对话文本
        if self.is_training:
            # 训练时需要构建完整的输入输出
            texts = []
            for target in target_texts:
                # 构建带有音频占位符的对话
                text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{user_prompt}"},
                        {"role": "assistant", "content": target}
                    ],
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
        else:
            # 推理时只需要输入
            texts = []
            for _ in target_texts:
                text = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{user_prompt}"}
                    ],
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
        
        # 使用processor处理音频和文本
        # 获取采样率（Qwen2-Audio 使用 16000Hz）
        sampling_rate = getattr(self.processor.feature_extractor, 'sampling_rate', 16000)
        
        try:
            inputs = self.processor(
                text=texts,
                audio=audios,  # 参数名是 audio 而非 audios
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"  Number of audios: {len(audios)}")
            print(f"  Audio shapes: {[a.shape for a in audios]}")
            print(f"  Sampling rate: {sampling_rate}")
            # 返回空批次
            return None
        
        # 添加labels用于训练
        if self.is_training:
            labels = inputs.input_ids.clone()
            # 将padding token设为-100
            labels[labels == self.tokenizer.pad_token_id] = -100
            inputs['labels'] = labels
        
        # 添加元信息
        inputs['ids'] = ids
        inputs['target_texts'] = target_texts
        inputs['audio_paths'] = audio_paths
        
        return inputs


def parse_triplets(text: str) -> List[tuple]:
    """
    解析三元组文本
    
    Args:
        text: 包含三元组的文本，格式为 <triplet> Subject <subj> Object <obj> Relation
        
    Returns:
        三元组列表 [(subject, object, relation), ...]
    """
    triplets = []
    
    # 按 <triplet> 分割
    parts = text.split('<triplet>')
    
    for part in parts[1:]:  # 跳过第一个空元素
        part = part.strip()
        if not part:
            continue
            
        try:
            # 解析 Subject <subj> Object <obj> Relation
            if '<subj>' in part and '<obj>' in part:
                # 提取subject
                subj_split = part.split('<subj>')
                subject = subj_split[0].strip()
                
                rest = subj_split[1] if len(subj_split) > 1 else ''
                
                # 提取object和relation
                obj_split = rest.split('<obj>')
                obj = obj_split[0].strip()
                relation = obj_split[1].strip() if len(obj_split) > 1 else ''
                
                # 清理relation（可能包含下一个triplet的开始）
                relation = relation.split('<triplet>')[0].strip()
                
                if subject and obj and relation:
                    triplets.append((subject, obj, relation))
        except Exception as e:
            print(f"Error parsing triplet: {part}, error: {e}")
            continue
    
    return triplets


def normalize_triplet(triplet: tuple) -> tuple:
    """
    标准化三元组
    
    Args:
        triplet: (subject, object, relation)
        
    Returns:
        标准化后的三元组
    """
    subject, obj, relation = triplet
    
    # 转小写并去除多余空格
    subject = ' '.join(subject.lower().split())
    obj = ' '.join(obj.lower().split())
    relation = relation.strip()
    
    return (subject, obj, relation)
