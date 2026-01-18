#!/bin/bash
# =============================================================================
# Qwen2-Audio 语音关系抽取 - 一键运行脚本
# =============================================================================

# ===== 配置区 =====
# 显卡设置 (多卡用逗号分隔，如 "0,1,2,3")
export CUDA_VISIBLE_DEVICES="0"

# SwanLab API Key (从 https://swanlab.cn 获取)
export SWANLAB_API_KEY="EcJUnP1993IKCvYXbXxJo"

# 模型路径
MODEL_PATH="/root/autodl-tmp/Qwen"

# 数据配置
DATA_DIR="./conll04"
DATASET="conll04"

# 输出目录
OUTPUT_DIR="./output"

# 训练超参数
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=7e-5
GRADIENT_ACCUMULATION=8

# SwanLab 项目配置
SWANLAB_PROJECT="qwen2-audio-re"
SWANLAB_EXPERIMENT="conll04-$(date +%Y%m%d_%H%M%S)"

# ===== 环境检查 =====
echo "=============================================="
echo "Qwen2-Audio 语音关系抽取训练"
echo "=============================================="
echo ""
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo ""

# 检查CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
echo ""

# 检查SwanLab Key
if [ "$SWANLAB_API_KEY" == "your_swanlab_api_key_here" ]; then
    echo "Warning: SwanLab API Key not set, logging will be disabled"
    NO_SWANLAB="--no-swanlab"
else
    echo "SwanLab Project: $SWANLAB_PROJECT"
    NO_SWANLAB=""
fi
echo ""

# ===== 开始训练 =====
echo "=============================================="
echo "Starting Training..."
echo "=============================================="

python main.py train \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    $NO_SWANLAB

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training Completed Successfully!"
    echo "=============================================="
    echo ""
    echo "Output files:"
    echo "  Best model: $OUTPUT_DIR/best_model/"
    echo "  Dev predictions: $OUTPUT_DIR/dev_predictions_epoch*.json"
    echo "  Test predictions: $OUTPUT_DIR/test_predictions.json"
    echo "  Test metrics: $OUTPUT_DIR/test_metrics.json"
    echo ""

    # 显示测试结果
    if [ -f "$OUTPUT_DIR/test_metrics.json" ]; then
        echo "Test Results:"
        cat "$OUTPUT_DIR/test_metrics.json"
        echo ""
    fi
else
    echo ""
    echo "=============================================="
    echo "Training Failed!"
    echo "=============================================="
    exit 1
fi
