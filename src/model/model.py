"""
模型模块
"""
import torch
from typing import Optional
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_processor(
    model_name_or_path: str,
    device: str = "cuda",
    use_lora: bool = True,
    lora_config: Optional[dict] = None,
    torch_dtype=torch.float16
):
    """
    加载Qwen2-Audio模型和处理器
    
    Args:
        model_name_or_path: 模型路径或名称
        device: 设备
        use_lora: 是否使用LoRA
        lora_config: LoRA配置
        torch_dtype: 数据类型
        
    Returns:
        model, processor, tokenizer
    """
    print(f"Loading model from {model_name_or_path}...")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # 加载模型
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 训练时不需要缓存，如果不关闭会有警告且可能影响梯度检查点
    model.config.use_cache = False
    
    # 获取tokenizer
    tokenizer = processor.tokenizer
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 应用LoRA
    if use_lora:
        model = apply_lora(model, lora_config)
        print("LoRA applied successfully!")
    
    # 打印可训练参数
    print_trainable_parameters(model)
    
    return model, processor, tokenizer


def apply_lora(model, lora_config: Optional[dict] = None):
    """
    应用LoRA到模型
    
    Args:
        model: 基础模型
        lora_config: LoRA配置字典
        
    Returns:
        应用LoRA后的模型
    """
    if lora_config is None:
        lora_config = {
            'r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.05,
            'target_modules': [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        }
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get('r', 64),
        lora_alpha=lora_config.get('lora_alpha', 128),
        lora_dropout=lora_config.get('lora_dropout', 0.05),
        target_modules=lora_config.get('target_modules', [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none"
    )
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 启用输入梯度（梯度检查点需要）
    model.enable_input_require_grads()
    
    # 应用PEFT
    model = get_peft_model(model, peft_config)
    
    return model


def print_trainable_parameters(model):
    """打印可训练参数数量"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def save_model(model, tokenizer, save_path: str):
    """
    保存模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        save_path: 保存路径
    """
    print(f"Saving model to {save_path}...")
    
    # 保存LoRA权重
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Model saved successfully!")


def load_trained_model(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    torch_dtype=torch.float16
):
    """
    加载训练好的模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA权重路径
        device: 设备
        torch_dtype: 数据类型
        
    Returns:
        model, processor, tokenizer
    """
    from peft import PeftModel
    
    print(f"Loading base model from {base_model_path}...")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 加载基础模型
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 加载LoRA权重
    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并权重（可选，用于推理加速）
    # model = model.merge_and_unload()
    
    tokenizer = processor.tokenizer
    
    print("Model loaded successfully!")
    
    return model, processor, tokenizer
