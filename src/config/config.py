"""
配置管理模块
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name_or_path: str = "Qwen/Qwen2-Audio-7B-Instruct"

    # 数据配置
    data_dir: str = "./conll04"
    train_file: str = "train_conll04.tsv"
    dev_file: str = "dev_conll04.tsv"
    test_file: str = "test_conll04.tsv"
    audio_dir: str = "./conll04/audio"
    dataset_name: str = "conll04"  # 用于选择模板

    # 训练超参数
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # LoRA 配置
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.06
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 输出配置
    output_dir: str = "./output"
    save_total_limit: int = 1

    # 日志配置
    logging_steps: int = 1
    logging_first_step: bool = True
    log_grad_norm: bool = True
    log_fractional_epoch: bool = True  # 像llamafactory一样记录0.1 epoch

    # SwanLab 配置
    use_swanlab: bool = True
    swanlab_project: str = "qwen2-audio-re"
    swanlab_experiment: str = "conll04-finetune"
    log_predictions: bool = True
    num_predictions_to_log: int = 5

    # 其他配置
    seed: int = 42
    bf16: bool = True  # 使用bf16，不需要混合精度
    max_length: int = 512

    # NEFTune 配置 (Embedding噪声)
    use_neftune: bool = True
    neftune_noise_alpha: float = 5.0

    # 解耦鲁棒性框架配置 (Decoupled Robustness Framework)
    # BCL - 边界感知对比学习 (Boundary-aware Contrastive Learning)
    use_bcl: bool = True
    lambda_bcl: float = 0.1  # BCL loss 权重
    bcl_margin: float = 0.5  # Margin Ranking Loss 的 margin 值
    
    # RDH - 反思性去幻觉 (Reflective De-Hallucination)
    use_rdh: bool = True
    lambda_rdh: float = 0.1  # RDH loss 权重

    # 设备配置
    device: str = "cuda"

    def __post_init__(self):
        """初始化后处理"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 构建完整路径
        self.train_path = os.path.join(self.data_dir, self.train_file)
        self.dev_path = os.path.join(self.data_dir, self.dev_file)
        self.test_path = os.path.join(self.data_dir, self.test_file)


@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "./output/best_model"
    batch_size: int = 18
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    device: str = "cuda"
    dataset_name: str = "conll04"
