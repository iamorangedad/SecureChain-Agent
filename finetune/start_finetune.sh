#!/bin/bash

# DeepSeek-Coder-1.3B 区块链代码微调启动脚本

echo "=========================================="
echo "DeepSeek-Coder-1.3B 区块链代码微调项目"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查CUDA
echo "检查CUDA环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠ 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "错误: 依赖安装失败"
    exit 1
fi

echo "✓ 依赖安装完成"

# 检查配置文件
if [ -f "config.json" ]; then
    echo "✓ 找到配置文件 config.json"
else
    echo "⚠ 未找到配置文件，将使用默认配置"
fi

# 询问是否使用GitHub Token
read -p "是否使用GitHub Token来提高API限制？(y/n): " use_token
if [ "$use_token" = "y" ] || [ "$use_token" = "Y" ]; then
    read -p "请输入GitHub Token: " github_token
    export GITHUB_TOKEN=$github_token
    echo "✓ GitHub Token已设置"
fi

# 询问运行模式
echo ""
echo "请选择运行模式:"
echo "1. 完整流程 (数据收集 -> 处理 -> 训练 -> 评估)"
echo "2. 仅数据收集"
echo "3. 仅数据处理"
echo "4. 仅模型训练"
echo "5. 仅模型评估"
echo "6. 交互式测试"

read -p "请输入选择 (1-6): " choice

case $choice in
    1)
        echo "运行完整流程..."
        python run_finetune.py --steps collect process train eval
        ;;
    2)
        echo "仅运行数据收集..."
        python run_finetune.py --steps collect
        ;;
    3)
        echo "仅运行数据处理..."
        python run_finetune.py --steps process
        ;;
    4)
        echo "仅运行模型训练..."
        python run_finetune.py --steps train
        ;;
    5)
        echo "仅运行模型评估..."
        python run_finetune.py --steps eval
        ;;
    6)
        echo "启动交互式测试..."
        python inference.py --mode interactive
        ;;
    *)
        echo "无效选择，运行完整流程..."
        python run_finetune.py --steps collect process train eval
        ;;
esac

echo ""
echo "=========================================="
echo "微调流程完成！"
echo "=========================================="

# 显示结果
if [ -f "finetune_report.json" ]; then
    echo "✓ 微调报告已生成: finetune_report.json"
fi

if [ -d "blockchain_coder_model" ]; then
    echo "✓ 模型已保存: blockchain_coder_model/"
fi

echo ""
echo "下一步操作:"
echo "1. 查看报告: cat finetune_report.json"
echo "2. 测试模型: python inference.py --mode interactive"
echo "3. 批量测试: python inference.py --mode batch"

echo ""
echo "感谢使用DeepSeek-Coder-1.3B区块链代码微调项目！" 