"""
数据处理模块
支持解耦鲁棒性框架 (Decoupled Robustness Framework)
- BCL: 边界感知对比学习 (Boundary-aware Contrastive Learning)
- RDH: 反思性去幻觉 (Reflective De-Hallucination)
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
import random
import re

from src.data.template import get_template, DatasetTemplate, RDH_REFLECTIVE_PROMPT


class AudioRECollator:
    """
    音频关系抽取数据整理器
    
    支持三种数据流:
    1. 标准 SFT: 常规监督微调
    2. BCL: 边界对比学习 (修改 label 获取负样本特征)
    3. RDH: 反思性去幻觉 (修改 instruction 注入错误实体)
    """
    
    def __init__(
        self,
        processor,
        tokenizer,
        template: DatasetTemplate,
        max_length: int = 512,
        is_training: bool = True,
        use_bcl: bool = True,
        use_rdh: bool = True
    ):
        """
        初始化整理器
        
        Args:
            processor: Qwen2-Audio处理器
            tokenizer: 分词器
            template: 数据集模板
            max_length: 最大序列长度
            is_training: 是否为训练模式
            use_bcl: 是否启用边界对比学习
            use_rdh: 是否启用反思性去幻觉
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.is_training = is_training
        self.use_bcl = use_bcl and is_training
        self.use_rdh = use_rdh and is_training
        
    def _extract_entities_from_triplets(self, target_text: str) -> List[str]:
        """
        从三元组文本中提取实体
        
        格式: <triplet> Subject <subj> Object <obj> Relation
        提取 Subject 和 Object
        """
        entities = []
        parts = target_text.split('<triplet>')
        
        for part in parts[1:]:  # 跳过第一个空元素
            part = part.strip()
            if not part:
                continue
            
            if '<subj>' in part and '<obj>' in part:
                # 提取 subject
                subj_split = part.split('<subj>')
                subject = subj_split[0].strip()
                if subject:
                    entities.append(subject)
                
                # 提取 object
                if len(subj_split) > 1:
                    obj_split = subj_split[1].split('<obj>')
                    obj = obj_split[0].strip()
                    if obj:
                        entities.append(obj)
        
        return entities
    
    def generate_boundary_negatives(self, entity: str) -> Tuple[str, str]:
        """
        生成边界负样本 (用于 BCL)
        
        Type A (Over): 添加上下文噪声 (e.g., "United States" -> "the United States")
        Type B (Under): 截取实体的一部分 (e.g., "United States" -> "States")
        
        Args:
            entity: 正确实体
            
        Returns:
            (over_boundary, under_boundary): 过边界和欠边界负样本
        """
        words = entity.split()
        
        # Type A: Over-boundary (添加前缀/后缀)
        prefixes = ["the", "a", "an", "Mr.", "Ms.", "Dr.", "Prof."]
        suffixes = ["'s", "Jr.", "Sr.", "Inc.", "Corp."]
        
        if random.random() > 0.5 and len(entity) > 2:
            # 添加前缀
            prefix = random.choice(prefixes)
            over_boundary = f"{prefix} {entity}"
        else:
            # 添加后缀
            suffix = random.choice(suffixes)
            over_boundary = f"{entity} {suffix}"
        
        # Type B: Under-boundary (截取部分)
        if len(words) > 1:
            # 对于多词实体，随机保留一部分
            if random.random() > 0.5:
                # 保留后半部分
                under_boundary = " ".join(words[len(words)//2:])
            else:
                # 保留前半部分
                under_boundary = " ".join(words[:len(words)//2 + 1])
        else:
            # 单词实体，截取一部分字符
            if len(entity) > 3:
                cut_point = random.randint(len(entity)//2, len(entity)-1)
                under_boundary = entity[:cut_point]
            else:
                under_boundary = entity  # 太短就不截取
        
        return over_boundary, under_boundary
    
    def generate_hallucination_negatives(self, entity: str) -> Tuple[str, str]:
        """
        生成幻觉负样本 (用于 RDH)
        
        Type C (Fabricated): 生造词 (e.g., "United States" -> "United States-ology")
        Type D (Phonetic): 音近错误词 (e.g., "Navigate" -> "Night rate")
        
        Args:
            entity: 正确实体
            
        Returns:
            (fabricated, phonetic): 生造词和音近词负样本
        """
        # Type C: Fabricated (添加假后缀)
        fake_suffixes = ["-ology", "-ism", "-tion", "-ness", "-ity", "-ment", "-ance"]
        fabricated = entity + random.choice(fake_suffixes)
        
        # Type D: Phonetic (音近替换)
        # 简单的字符替换模拟音近错误
        phonetic_map = {
            'a': 'e', 'e': 'i', 'i': 'a', 'o': 'u', 'u': 'o',
            's': 'z', 'c': 'k', 'k': 'c', 'n': 'm', 'm': 'n',
            't': 'd', 'd': 't', 'b': 'p', 'p': 'b', 'g': 'j',
        }
        
        phonetic_chars = list(entity.lower())
        # 随机替换1-2个字符
        num_changes = min(random.randint(1, 2), len(phonetic_chars))
        change_indices = random.sample(range(len(phonetic_chars)), num_changes)
        
        for idx in change_indices:
            char = phonetic_chars[idx]
            if char in phonetic_map:
                phonetic_chars[idx] = phonetic_map[char]
        
        phonetic = "".join(phonetic_chars)
        
        # 保持首字母大写（如果原来是大写的话）
        if entity[0].isupper():
            phonetic = phonetic.capitalize()
        
        return fabricated, phonetic
    
    def _build_rdh_text(self, system_prompt: str, noisy_entity: str, correct_target: str) -> str:
        """
        构建 RDH 流的文本 (Reflective Prompt)
        
        Input: 带有错误实体的 Reflective Prompt
        Label: 正确实体 (保持不变)
        """
        rdh_user_prompt = RDH_REFLECTIVE_PROMPT.format(noisy_entity=noisy_entity)
        
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<|audio_bos|><|AUDIO|><|audio_eos|>\n{rdh_user_prompt}"},
                {"role": "assistant", "content": correct_target}
            ],
            tokenize=False,
            add_generation_prompt=False
        )
        return text
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 批次数据列表
            
        Returns:
            整理后的批次数据，包含 SFT、BCL、RDH 三种流的数据
        """
        ids = [item['id'] for item in batch]
        audios = [item['audio'].numpy() for item in batch]
        target_texts = [item['target_text'] for item in batch]
        audio_paths = [item['audio_path'] for item in batch]
        
        # 获取模板内容
        system_prompt = self.template.get_full_system_prompt()
        user_prompt = self.template.user_prompt
        
        # ============== 流 1: 标准 SFT ==============
        if self.is_training:
            texts = []
            for target in target_texts:
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
        sampling_rate = getattr(self.processor.feature_extractor, 'sampling_rate', 16000)
        
        try:
            inputs = self.processor(
                text=texts,
                audio=audios,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"Error processing batch: {e}")
            print(f"  Number of audios: {len(audios)}")
            print(f"  Audio shapes: {[a.shape for a in audios]}")
            print(f"  Sampling rate: {sampling_rate}")
            return None
        
        # 添加labels用于训练
        if self.is_training:
            labels = inputs.input_ids.clone()
            
            # Mask prompt 部分，只对 assistant 回复计算 loss
            for i in range(labels.size(0)):
                input_ids = inputs.input_ids[i].tolist()
                
                try:
                    text_decoded = self.tokenizer.decode(input_ids)
                    assistant_marker = "<|im_start|>assistant\n"
                    
                    idx = text_decoded.rfind(assistant_marker)
                    if idx != -1:
                        prompt_text = text_decoded[:idx + len(assistant_marker)]
                        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                        prompt_length = len(prompt_tokens)
                        labels[i, :prompt_length] = -100
                    else:
                        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                        assistant_start = -1
                        for j in range(len(input_ids) - 1):
                            if input_ids[j] == im_start_id:
                                next_tokens = self.tokenizer.decode([input_ids[j+1]] if j+1 < len(input_ids) else [])
                                if 'assistant' in next_tokens.lower():
                                    assistant_start = j
                        
                        if assistant_start != -1:
                            content_start = assistant_start + 3
                            labels[i, :content_start] = -100
                        else:
                            seq_len = (labels[i] != self.tokenizer.pad_token_id).sum().item()
                            labels[i, :seq_len // 2] = -100
                except Exception as e:
                    seq_len = (labels[i] != self.tokenizer.pad_token_id).sum().item()
                    prompt_len = int(seq_len * 0.6)
                    labels[i, :prompt_len] = -100
            
            labels[labels == self.tokenizer.pad_token_id] = -100
            inputs['labels'] = labels
        
        # ============== 流 2: BCL (边界对比学习) ==============
        if self.use_bcl and self.is_training:
            bcl_pos_entities = []  # 正确实体
            bcl_neg_entities = []  # 错误边界实体
            
            for target in target_texts:
                entities = self._extract_entities_from_triplets(target)
                if entities:
                    # 随机选择一个实体
                    entity = random.choice(entities)
                    bcl_pos_entities.append(entity)
                    
                    # 生成边界负样本
                    over_boundary, under_boundary = self.generate_boundary_negatives(entity)
                    # 随机选择 over 或 under
                    neg_entity = random.choice([over_boundary, under_boundary])
                    bcl_neg_entities.append(neg_entity)
                else:
                    # 如果没有提取到实体，使用整个 target 作为正样本
                    bcl_pos_entities.append(target)
                    bcl_neg_entities.append(target + "-noise")
            
            inputs['bcl_pos_entities'] = bcl_pos_entities
            inputs['bcl_neg_entities'] = bcl_neg_entities
        
        # ============== 流 3: RDH (反思性去幻觉) ==============
        if self.use_rdh and self.is_training:
            rdh_texts = []
            rdh_correct_entities = []  # 正确答案 (用于 label)
            
            for target in target_texts:
                entities = self._extract_entities_from_triplets(target)
                if entities:
                    # 随机选择一个实体
                    entity = random.choice(entities)
                    
                    # 生成幻觉负样本
                    fabricated, phonetic = self.generate_hallucination_negatives(entity)
                    # 随机选择一种错误类型
                    noisy_entity = random.choice([fabricated, phonetic])
                    
                    # 构建 RDH 文本 (注入错误实体到 instruction)
                    rdh_text = self._build_rdh_text(system_prompt, noisy_entity, entity)
                    rdh_texts.append(rdh_text)
                    rdh_correct_entities.append(entity)
                else:
                    # 如果没有提取到实体，使用简化的 RDH
                    rdh_text = self._build_rdh_text(system_prompt, "unknown entity", target)
                    rdh_texts.append(rdh_text)
                    rdh_correct_entities.append(target)
            
            # 处理 RDH 输入
            try:
                rdh_inputs = self.processor(
                    text=rdh_texts,
                    audio=audios,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=True,
                )
                
                # 为 RDH 创建 labels
                rdh_labels = rdh_inputs.input_ids.clone()
                
                for i in range(rdh_labels.size(0)):
                    input_ids = rdh_inputs.input_ids[i].tolist()
                    
                    try:
                        text_decoded = self.tokenizer.decode(input_ids)
                        assistant_marker = "<|im_start|>assistant\n"
                        
                        idx = text_decoded.rfind(assistant_marker)
                        if idx != -1:
                            prompt_text = text_decoded[:idx + len(assistant_marker)]
                            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                            prompt_length = len(prompt_tokens)
                            rdh_labels[i, :prompt_length] = -100
                        else:
                            seq_len = (rdh_labels[i] != self.tokenizer.pad_token_id).sum().item()
                            rdh_labels[i, :seq_len // 2] = -100
                    except Exception:
                        seq_len = (rdh_labels[i] != self.tokenizer.pad_token_id).sum().item()
                        prompt_len = int(seq_len * 0.6)
                        rdh_labels[i, :prompt_len] = -100
                
                rdh_labels[rdh_labels == self.tokenizer.pad_token_id] = -100
                
                inputs['rdh_input_ids'] = rdh_inputs.input_ids
                inputs['rdh_attention_mask'] = rdh_inputs.attention_mask
                inputs['rdh_labels'] = rdh_labels
                if 'input_features' in rdh_inputs:
                    inputs['rdh_input_features'] = rdh_inputs.input_features
                if 'feature_attention_mask' in rdh_inputs:
                    inputs['rdh_feature_attention_mask'] = rdh_inputs.feature_attention_mask
                inputs['rdh_correct_entities'] = rdh_correct_entities
                
            except Exception as e:
                print(f"Warning: Error processing RDH batch: {e}")
                # RDH 处理失败时禁用该流
                inputs['rdh_input_ids'] = None
        
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
