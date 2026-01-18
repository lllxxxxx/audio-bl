"""
数据模板模块
支持不同数据集的配置
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# RDH (Reflective De-Hallucination) 专用模板
# 用于训练模型纠正错误实体
RDH_REFLECTIVE_PROMPT = 'I detected a potential error in the extracted entity: "{noisy_entity}". Based on the audio, check and output the corrected entity string.'


@dataclass
class DatasetTemplate:
    """数据集模板基类"""
    name: str
    # 关系类型列表
    relation_types: List[str]
    # 系统提示词
    system_prompt: str
    # 用户提示词模板
    user_prompt: str
    # 输出格式说明
    output_format: str
    # TSV列名映射
    id_column: str = "id"
    audio_column: str = "audio"
    target_column: str = "tgt_text"
    
    def get_full_system_prompt(self) -> str:
        """获取完整的系统提示词"""
        relation_list = "、".join(self.relation_types)
        return f"{self.system_prompt}\n关系类型包括：{relation_list}。\n{self.output_format}"


# CoNLL04 数据集模板
CONLL04_TEMPLATE = DatasetTemplate(
    name="conll04",
    relation_types=[
        "OrgBased_In",   # 组织位于
        "Work_For",      # 为...工作
        "Located_In",    # 位于
        "Live_In",       # 居住在
        "Kill"           # 杀死
    ],
    system_prompt="你是一个语音关系抽取助手。请从给定的音频中识别并提取实体之间的关系三元组。",
    user_prompt="请从上述音频中抽取所有关系三元组。",
    output_format="请按照以下格式输出：<triplet> 主体 <subj> 客体 <obj> 关系类型\n如果有多个三元组，请依次输出。"
)


# NYT 数据集模板 (示例)
NYT_TEMPLATE = DatasetTemplate(
    name="nyt",
    relation_types=[
        "/people/person/nationality",
        "/location/location/contains",
        "/people/person/place_lived",
        "/people/person/place_of_birth",
        "/business/person/company",
        "/people/person/religion",
        "/location/country/capital",
        "/location/neighborhood/neighborhood_of",
        "/business/company/founders",
        "/people/person/ethnicity",
        "/people/person/children",
        "/location/country/administrative_divisions",
        "/people/deceased_person/place_of_death",
        "/business/company/place_founded",
        "/location/us_state/capital"
    ],
    system_prompt="你是一个语音关系抽取助手。请从给定的音频中识别并提取实体之间的关系三元组。",
    user_prompt="请从上述音频中抽取所有关系三元组。",
    output_format="请按照以下格式输出：<triplet> 主体 <subj> 客体 <obj> 关系类型\n如果有多个三元组，请依次输出。"
)


# WebNLG 数据集模板 (示例)
WEBNLG_TEMPLATE = DatasetTemplate(
    name="webnlg",
    relation_types=[
        "country",
        "city_served",
        "runway_length",
        "elevation",
        "leader",
        "architect",
        "owner",
        "location",
        "operator",
        "publisher"
        # 可扩展更多
    ],
    system_prompt="你是一个语音关系抽取助手。请从给定的音频中识别并提取实体之间的关系三元组。",
    user_prompt="请从上述音频中抽取所有关系三元组。",
    output_format="请按照以下格式输出：<triplet> 主体 <subj> 客体 <obj> 关系类型\n如果有多个三元组，请依次输出。"
)


# 模板注册表
DATASET_TEMPLATES: Dict[str, DatasetTemplate] = {
    "conll04": CONLL04_TEMPLATE,
    "nyt": NYT_TEMPLATE,
    "webnlg": WEBNLG_TEMPLATE,
}


def get_template(dataset_name: str) -> DatasetTemplate:
    """
    获取数据集模板
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        对应的数据集模板
    """
    if dataset_name not in DATASET_TEMPLATES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_TEMPLATES.keys())}")
    return DATASET_TEMPLATES[dataset_name]


def register_template(template: DatasetTemplate):
    """
    注册新的数据集模板
    
    Args:
        template: 数据集模板
    """
    DATASET_TEMPLATES[template.name] = template


def create_custom_template(
    name: str,
    relation_types: List[str],
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    output_format: Optional[str] = None
) -> DatasetTemplate:
    """
    创建自定义数据集模板
    
    Args:
        name: 数据集名称
        relation_types: 关系类型列表
        system_prompt: 系统提示词 (可选)
        user_prompt: 用户提示词 (可选)
        output_format: 输出格式 (可选)
        
    Returns:
        自定义的数据集模板
    """
    template = DatasetTemplate(
        name=name,
        relation_types=relation_types,
        system_prompt=system_prompt or "你是一个语音关系抽取助手。请从给定的音频中识别并提取实体之间的关系三元组。",
        user_prompt=user_prompt or "请从上述音频中抽取所有关系三元组。",
        output_format=output_format or "请按照以下格式输出：<triplet> 主体 <subj> 客体 <obj> 关系类型\n如果有多个三元组，请依次输出。"
    )
    
    # 自动注册
    register_template(template)
    
    return template
