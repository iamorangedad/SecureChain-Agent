# DeepSeek-Coder-1.3B 区块链代码微调项目

这个项目基于DeepSeek-Coder-1.3B模型，专门针对区块链智能合约代码进行微调，重点提升模型在安全审计、漏洞检测和代码修复方面的能力。

## 项目结构

```
finetune/
├── requirements.txt          # 依赖包
├── data_processor.py         # 数据预处理脚本
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── run_finetune.py          # 完整流程脚本
├── README.md                # 项目说明
├── processed_data/          # 处理后的数据
├── blockchain_coder_model/  # 微调后的模型
└── logs/                    # 训练日志
```

## 功能特性

### 🎯 核心功能
- **安全审计**: 自动识别智能合约中的安全漏洞
- **漏洞修复**: 提供具体的修复建议和代码
- **代码解释**: 详细解释合约功能和潜在风险
- **最佳实践**: 提供区块链开发最佳实践建议

### 🔧 技术特性
- **LoRA微调**: 使用LoRA技术进行高效微调
- **4bit量化**: 支持4bit量化减少显存占用
- **多语言支持**: 支持Solidity、Rust、Vyper
- **漏洞案例**: 包含20+个真实漏洞案例

## 快速开始

### 1. 安装依赖

```bash
cd finetune
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 运行完整的微调流程（数据收集 -> 处理 -> 训练 -> 评估）
python run_finetune.py

# 或者分步运行
python run_finetune.py --steps collect process train eval
```

### 3. 使用GitHub Token（推荐）

```bash
python run_finetune.py --github_token YOUR_GITHUB_TOKEN
```

## 详细使用说明

### 数据收集

项目会自动从GitHub收集智能合约代码：

```bash
# 收集通用合约数据
python ../smart_contract_collector.py --output collected_contracts

# 收集漏洞案例
python ../vulnerability_collector.py --output vulnerability_cases
```

### 数据处理

```bash
python data_processor.py
```

处理后的数据将保存在 `processed_data/blockchain_dataset/` 目录中。

### 模型训练

```bash
python train.py
```

训练配置：
- **基础模型**: deepseek-ai/deepseek-coder-1.3b-base
- **微调方法**: LoRA (Low-Rank Adaptation)
- **量化**: 4bit量化
- **训练轮数**: 3 epochs
- **学习率**: 2e-4
- **批次大小**: 4 (梯度累积4步)

### 模型推理

```bash
# 交互式测试
python inference.py --mode interactive

# 批量测试
python inference.py --mode batch --test_file test_cases.json

# 创建测试案例
python inference.py --create_test
```

## 训练配置

### 硬件要求

**最低配置**:
- GPU: 8GB显存 (RTX 3070或同等)
- RAM: 16GB
- 存储: 20GB可用空间

**推荐配置**:
- GPU: 16GB+显存 (RTX 4080/4090或同等)
- RAM: 32GB
- 存储: 50GB可用空间

### 训练参数

可以在 `train.py` 中修改 `TrainingConfig` 类来调整训练参数：

```python
@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
```

## 数据集说明

### 数据来源
1. **GitHub智能合约**: 从GitHub收集的Solidity、Rust、Vyper代码
2. **已知漏洞案例**: 20+个真实的安全漏洞案例
3. **漏洞模式**: 8种常见漏洞模式的代码样本

### 数据统计
- **训练样本**: 根据收集的数据量动态生成
- **验证样本**: 10%的数据
- **测试样本**: 10%的数据

### 数据格式
```json
{
    "instruction": "请对以下Solidity智能合约进行安全审计...",
    "input": "",
    "output": "经过安全审计，发现以下潜在漏洞...",
    "language": "solidity",
    "type": "security_audit"
}
```

## 模型性能

### 评估指标
- **损失函数**: Cross-Entropy Loss
- **评估指标**: eval_loss
- **早停策略**: 验证损失连续3次不下降时停止

### 预期效果
经过微调后，模型应该能够：
1. 准确识别常见的安全漏洞
2. 提供具体的修复建议
3. 生成安全的代码示例
4. 解释代码的功能和风险

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   batch_size: int = 2
   gradient_accumulation_steps: int = 8
   ```

2. **GitHub API限制**
   ```bash
   # 使用GitHub Token
   python run_finetune.py --github_token YOUR_TOKEN
   ```

3. **依赖安装失败**
   ```bash
   # 使用conda安装PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

4. **数据集为空**
   ```bash
   # 检查数据收集是否成功
   ls collected_contracts/solidity/
   ls vulnerability_cases/
   ```

### 日志查看

训练日志保存在 `logs/` 目录中，可以使用TensorBoard查看：

```bash
tensorboard --logdir logs/
```

## 高级用法

### 自定义训练

1. **修改数据处理器**
   ```python
   # 在 data_processor.py 中添加新的提示词模板
   self.prompts['solidity']['custom_task'] = "自定义任务描述"
   ```

2. **调整LoRA配置**
   ```python
   # 在 train.py 中修改LoRA参数
   lora_config = LoraConfig(
       r=32,  # 增加rank
       lora_alpha=64,
       target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
   )
   ```

3. **使用自定义数据集**
   ```python
   # 准备自定义数据
   custom_data = [
       {"instruction": "...", "input": "", "output": "..."}
   ]
   ```

### 模型部署

训练完成后，模型可以用于：

1. **API服务**
   ```python
   from inference import BlockchainCodeInference
   
   inference = BlockchainCodeInference("blockchain_coder_model")
   result = inference.security_audit(code)
   ```

2. **Web界面**
   ```python
   # 使用Gradio创建Web界面
   import gradio as gr
   
   def audit_code(code):
       return inference.security_audit(code)
   
   gr.Interface(fn=audit_code, inputs="text", outputs="text").launch()
   ```

## 安全提醒

⚠️ **重要警告**

- 微调后的模型仅用于安全研究和学习目的
- 请勿在生产环境中直接使用模型生成的代码
- 建议结合专业的安全审计工具进行验证
- 遵守相关法律法规和道德准则

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目仅用于教育和研究目的。请遵守相关法律法规。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 这个项目需要大量的计算资源和时间。建议在有GPU的环境中运行，并确保有足够的存储空间。 