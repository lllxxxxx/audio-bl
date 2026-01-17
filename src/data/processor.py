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
            
            # 关键修复：不仅mask padding，还要mask掉prompt部分
            # 只对assistant的回复计算loss
            for i in range(labels.size(0)):
                input_ids = inputs.input_ids[i].tolist()
                
                # 找到assistant回复的起始位置
                # Qwen2的格式是 <|im_start|>assistant\n ... <|im_end|>
                # 需要找到 "assistant" 后的内容开始位置
                
                # 方法1：找 <|im_start|>assistant 的位置
                # assistant token的id
                try:
                    # 尝试找assistant标记
                    # Qwen2格式: ...<|im_start|>assistant\n{content}<|im_end|>
                    text_decoded = self.tokenizer.decode(input_ids)
                    
                    # 找到最后一个 assistant 出现的位置（因为可能有多轮对话）
                    # 我们要找 <|im_start|>assistant\n 之后的内容
                    assistant_marker = "<|im_start|>assistant\n"
                    
                    # 先找到assistant在原始text中的位置
                    idx = text_decoded.rfind(assistant_marker)
                    if idx != -1:
                        # 计算 assistant marker 之前的文本
                        prompt_text = text_decoded[:idx + len(assistant_marker)]
                        # 编码来获取token数量
                        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                        prompt_length = len(prompt_tokens)
                        
                        # mask掉prompt部分
                        labels[i, :prompt_length] = -100
                    else:
                        # 如果找不到assistant标记，使用备用方法
                        # 找 <|im_start|>assistant 的token序列
                        # im_start token id
                        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
                        
                        # 遍历找到最后一个 im_start 后跟 assistant 的位置
                        assistant_start = -1
                        for j in range(len(input_ids) - 1):
                            if input_ids[j] == im_start_id:
                                # 检查后面是否是 assistant
                                next_tokens = self.tokenizer.decode([input_ids[j+1]] if j+1 < len(input_ids) else [])
                                if 'assistant' in next_tokens.lower():
                                    assistant_start = j
                        
                        if assistant_start != -1:
                            # 找到assistant\n后的第一个内容token
                            # 通常是 im_start, assistant, \n, 然后是内容
                            content_start = assistant_start + 3  # 跳过 im_start, assistant, \n
                            labels[i, :content_start] = -100
                        else:
                            # 最后的备用方案：只mask掉一半（假设前一半是prompt）
                            seq_len = (labels[i] != self.tokenizer.pad_token_id).sum().item()
                            labels[i, :seq_len // 2] = -100
                    
                except Exception as e:
                    # 出错时使用简单的启发式方法
                    # 假设prompt占序列的前60%
                    seq_len = (labels[i] != self.tokenizer.pad_token_id).sum().item()
                    prompt_len = int(seq_len * 0.6)
                    labels[i, :prompt_len] = -100
            
            # 仍然要mask padding token
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
