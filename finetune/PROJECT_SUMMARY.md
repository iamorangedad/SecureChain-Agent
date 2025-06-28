# DeepSeek-Coder-1.3B 区块链代码微调项目总结

## 项目概述

本项目基于DeepSeek-Coder-1.3B模型，专门针对区块链智能合约代码进行微调，重点提升模型在安全审计、漏洞检测和代码修复方面的能力。项目包含完整的数据收集、处理、训练和推理流程。

## 项目架构

```
项目根目录/
├── smart_contract_collector.py      # 通用智能合约收集器
├── vulnerability_collector.py       # 漏洞案例收集器
├── requirements.txt                 # 基础依赖
├── README.md                       # 项目说明
└── finetune/                       # 微调项目目录
    ├── requirements.txt            # 微调依赖
    ├── data_processor.py           # 数据预处理
    ├── train.py                    # 训练脚本
    ├── inference.py                # 推理脚本
    ├── run_finetune.py            # 完整流程脚本
    ├── start_finetune.sh          # 启动脚本
    ├── config.json                # 配置文件
    ├── README.md                  # 微调说明
    └── PROJECT_SUMMARY.md         # 项目总结
```

## 核心功能

### 1. 数据收集模块
- **智能合约收集器**: 从GitHub收集Solidity、Rust、Vyper代码
- **漏洞案例收集器**: 收集20+个真实的安全漏洞案例
- **支持GitHub API**: 可配置Token提高API限制

### 2. 数据处理模块
- **代码清理**: 移除注释和空行
- **样本生成**: 生成4种类型的训练样本
  - 安全审计 (security_audit)
  - 漏洞修复 (vulnerability_fix)
  - 代码解释 (code_explanation)
  - 最佳实践 (best_practices)
- **数据集分割**: 自动分割为训练/验证/测试集

### 3. 模型训练模块
- **基础模型**: deepseek-ai/deepseek-coder-1.3b-base
- **微调方法**: LoRA (Low-Rank Adaptation)
- **量化技术**: 4bit量化减少显存占用
- **训练优化**: 梯度检查点、混合精度训练

### 4. 推理评估模块
- **交互式测试**: 实时输入代码进行测试
- **批量测试**: 批量处理测试案例
- **多语言支持**: Solidity、Rust、Vyper

## 技术特性

### 模型配置
```python
# 基础模型
model_name: "deepseek-ai/deepseek-coder-1.3b-base"

# LoRA配置
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# 训练配置
num_epochs: 3
batch_size: 4
learning_rate: 2e-4
max_seq_length: 2048
```

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

## 使用方法

### 快速开始

1. **安装依赖**
   ```bash
   cd finetune
   pip install -r requirements.txt
   ```

2. **运行完整流程**
   ```bash
   # 使用启动脚本
   ./start_finetune.sh
   
   # 或直接运行
   python run_finetune.py
   ```

3. **分步运行**
   ```bash
   # 仅数据收集
   python run_finetune.py --steps collect
   
   # 仅数据处理
   python run_finetune.py --steps process
   
   # 仅模型训练
   python run_finetune.py --steps train
   
   # 仅模型评估
   python run_finetune.py --steps eval
   ```

### 使用GitHub Token

```bash
python run_finetune.py --github_token YOUR_GITHUB_TOKEN
```

### 自定义配置

1. **修改配置文件**
   ```bash
   # 编辑 config.json
   vim config.json
   ```

2. **使用自定义配置**
   ```bash
   python run_finetune.py --config config.json
   ```

## 训练流程

### 1. 数据收集阶段
- 从GitHub搜索智能合约代码
- 收集已知漏洞案例
- 生成元数据文件

### 2. 数据处理阶段
- 清理和标准化代码
- 生成训练样本
- 分割数据集

### 3. 模型训练阶段
- 加载预训练模型
- 应用LoRA适配器
- 执行微调训练

### 4. 模型评估阶段
- 创建测试案例
- 批量评估模型
- 生成评估报告

## 输出文件

### 数据文件
```
collected_contracts/
├── solidity/
│   ├── metadata.json
│   └── *.sol
├── rust/
│   ├── metadata.json
│   └── *.rs
└── vyper/
    ├── metadata.json
    └── *.vy

vulnerability_cases/
├── solidity/
├── rust/
├── all_vulnerability_cases.json
└── vulnerability_report.md
```

### 模型文件
```
blockchain_coder_model/
├── adapter_config.json
├── adapter_model.bin
├── training_config.json
├── eval_results.json
└── tokenizer files
```

### 报告文件
```
finetune_report.json          # 微调报告
test_results_*.json          # 测试结果
dataset_stats.json           # 数据集统计
```

## 性能指标

### 硬件要求
- **最低配置**: 8GB显存，16GB内存
- **推荐配置**: 16GB+显存，32GB内存

### 训练时间
- **数据收集**: 30-60分钟（取决于网络和API限制）
- **数据处理**: 5-10分钟
- **模型训练**: 2-6小时（取决于硬件配置）

### 预期效果
- 准确识别常见安全漏洞
- 提供具体的修复建议
- 生成安全的代码示例
- 解释代码功能和风险

## 安全考虑

### 数据安全
- 仅收集公开的GitHub代码
- 不包含敏感信息
- 遵守GitHub使用条款

### 模型安全
- 仅用于安全研究和学习
- 不用于生产环境
- 建议结合专业工具验证

### 使用限制
- 遵守相关法律法规
- 遵循道德准则
- 不得用于恶意目的

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 增加gradient_accumulation_steps
   - 使用更小的模型

2. **GitHub API限制**
   - 使用GitHub Token
   - 减少搜索频率
   - 等待限制重置

3. **依赖安装失败**
   - 使用conda安装PyTorch
   - 检查Python版本兼容性
   - 更新pip和setuptools

4. **训练中断**
   - 检查磁盘空间
   - 监控GPU温度
   - 使用checkpoint恢复

### 调试技巧

1. **查看日志**
   ```bash
   tail -f logs/training.log
   ```

2. **监控资源**
   ```bash
   nvidia-smi -l 1
   htop
   ```

3. **检查数据**
   ```bash
   python -c "from datasets import load_from_disk; ds = load_from_disk('processed_data/blockchain_dataset'); print(ds)"
   ```

## 扩展功能

### 自定义数据
- 添加新的编程语言支持
- 扩展漏洞类型
- 自定义提示词模板

### 模型优化
- 尝试不同的LoRA配置
- 调整训练参数
- 使用更大的基础模型

### 部署方案
- API服务部署
- Web界面集成
- 移动端应用

## 贡献指南

### 开发环境
1. Fork项目
2. 创建特性分支
3. 编写代码和测试
4. 提交Pull Request

### 代码规范
- 遵循PEP 8
- 添加类型注解
- 编写文档字符串
- 包含单元测试

### 提交规范
- 使用清晰的提交信息
- 关联Issue编号
- 描述功能变更

## 许可证

本项目仅用于教育和研究目的。请遵守相关法律法规。

## 联系方式

- GitHub Issues: 提交问题和建议
- 邮件联系: 项目维护者邮箱
- 文档更新: 定期更新使用说明

---

**注意**: 本项目需要大量计算资源和时间。建议在有GPU的环境中运行，并确保有足够的存储空间。微调后的模型仅用于安全研究和学习目的，请勿在生产环境中直接使用。 