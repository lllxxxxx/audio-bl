"""
数据集模块
"""
import os
import csv
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
import librosa

from src.data.template import get_template, DatasetTemplate


class AudioREDataset(Dataset):
    """语音关系抽取数据集"""
    
    def __init__(
        self,
        tsv_path: str,
        audio_base_dir: str,
        processor,
        tokenizer,
        template: DatasetTemplate,
        max_length: int = 512,
        is_training: bool = True,
        sample_rate: int = 16000
    ):
        """
        初始化数据集
        
        Args:
            tsv_path: TSV文件路径
            audio_base_dir: 音频文件基础目录
            processor: Qwen2-Audio处理器
            tokenizer: 分词器
            template: 数据集模板
            max_length: 最大序列长度
            is_training: 是否为训练模式
            sample_rate: 音频采样率
        """
        self.tsv_path = tsv_path
        self.audio_base_dir = audio_base_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = max_length
        self.is_training = is_training
        self.sample_rate = sample_rate
        
        # 加载数据
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载TSV数据"""
        data = []
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # 解析音频路径
                audio_info = row[self.template.audio_column]
                # 格式: /path/to/audio.wav:start:end
                audio_path = audio_info.split(':')[0]
                
                # 确定实际音频路径
                # 从原始路径提取文件名
                audio_filename = os.path.basename(audio_path)
                # 确定split (train/dev/test)
                sample_id = row[self.template.id_column]
                if 'train' in sample_id:
                    split = 'train'
                elif 'dev' in sample_id:
                    split = 'dev'
                else:
                    split = 'test'
                
                actual_audio_path = os.path.join(self.audio_base_dir, split, audio_filename)
                
                data.append({
                    'id': sample_id,
                    'audio_path': actual_audio_path,
                    'target_text': row[self.template.target_column],
                    'speaker': row.get('speaker', '0'),
                    'language': row.get('tgt_lang', 'en')
                })
        
        return data
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return torch.tensor(audio)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 返回空音频
            return torch.zeros(self.sample_rate)  # 1秒静音
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        item = self.data[idx]
        
        # 加载音频
        audio = self._load_audio(item['audio_path'])
        
        # 使用模板构建对话格式
        system_prompt = self.template.get_full_system_prompt()
        user_prompt = self.template.user_prompt
        
        conversation = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio.numpy()},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        if self.is_training:
            # 训练时添加assistant回复
            conversation.append({
                "role": "assistant",
                "content": item['target_text']
            })
        
        return {
            'id': item['id'],
            'conversation': conversation,
            'audio': audio,
            'target_text': item['target_text'],
            'audio_path': item['audio_path']
        }


def create_dataloaders(
    config,
    processor,
    tokenizer,
    train_collator,
    eval_collator
):
    """创建数据加载器"""
    from torch.utils.data import DataLoader
    
    # 获取模板
    template = get_template(config.dataset_name)
    
    # 创建数据集
    train_dataset = AudioREDataset(
        tsv_path=config.train_path,
        audio_base_dir=config.audio_dir,
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=config.max_length,
        is_training=True
    )
    
    dev_dataset = AudioREDataset(
        tsv_path=config.dev_path,
        audio_base_dir=config.audio_dir,
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=config.max_length,
        is_training=False
    )
    
    test_dataset = AudioREDataset(
        tsv_path=config.test_path,
        audio_base_dir=config.audio_dir,
        processor=processor,
        tokenizer=tokenizer,
        template=template,
        max_length=config.max_length,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=0,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader
