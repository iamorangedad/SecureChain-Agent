# SecureChain Agent

A specialized AI-powered coding agent designed for blockchain development, focusing on smart contract security auditing, vulnerability detection, and code generation. Built on DeepSeek-Coder-1.3B with fine-tuning for blockchain-specific tasks.

## 🎯 Project Overview

SecureChain Agent is an intelligent coding assistant that helps blockchain developers write secure and reliable smart contract code. It automatically detects potential vulnerabilities (such as reentrancy attacks, integer overflow), provides修复建议, and generates secure code templates following best practices.

### Key Features

- 🔍 **Automated Security Auditing**: Detect common vulnerabilities in smart contracts
- 🛠️ **Vulnerability Fixing**: Provide specific repair suggestions and optimized code
- 📝 **Secure Code Generation**: Generate code templates following security best practices
- 🔄 **Real-time Code Review**: Integrated security auditing during development
- 🌐 **Multi-language Support**: Solidity (Ethereum), Rust (Solana/Polkadot), Vyper
- 🧪 **Test Generation**: Automatically generate unit tests for smart contracts

## 🏗️ Architecture

```
SecureChain Agent/
├── data/                          # Data collection tools
│   ├── smart_contract_collector.py    # Smart contract code collector
│   ├── vulnerability_collector.py     # Vulnerability case collector
│   └── README.md                      # Data collection documentation
├── finetune/                      # Model fine-tuning
│   ├── data_processor.py             # Data preprocessing
│   ├── train.py                      # Training script
│   ├── inference.py                  # Inference script
│   ├── run_finetune.py              # Complete workflow
│   ├── start_finetune.sh            # Startup script
│   ├── config.json                  # Configuration
│   └── README.md                    # Fine-tuning documentation
└── Blockchain_Coding_Agent_Design.markdown  # Project design document
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ RAM
- 20GB+ available storage

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SecureChain-Agent
   ```

2. **Install dependencies**
   ```bash
   # Install data collection dependencies
   cd data
   pip install -r requirements.txt
   
   # Install fine-tuning dependencies
   cd ../finetune
   pip install -r requirements.txt
   ```

3. **Run the complete workflow**
   ```bash
   # Start the complete fine-tuning process
   ./start_finetune.sh
   
   # Or run step by step
   python run_finetune.py
   ```

### Using GitHub Token (Recommended)

For better data collection performance, use a GitHub Personal Access Token:

```bash
python run_finetune.py --github_token YOUR_GITHUB_TOKEN
```

## 📊 Core Capabilities

### 1. Vulnerability Detection

The agent can identify common smart contract vulnerabilities:

- **Reentrancy Attacks**: Detect unprotected external calls
- **Integer Overflow/Underflow**: Check for unsafe arithmetic operations
- **Access Control Issues**: Identify unprotected functions
- **Unchecked Return Values**: Detect unverified external call results
- **Gas Optimization Problems**: Identify high gas consumption patterns
- **Timestamp Dependencies**: Detect logic dependent on `block.timestamp`

### 2. Code Repair Suggestions

Automatically generate fix suggestions with examples:

```solidity
// Original vulnerable code
function withdraw(uint amount) public {
    require(balance[msg.sender] >= amount);
    (bool success, ) = msg.sender.call{value: amount}("");
    balance[msg.sender] -= amount;
}

// Fixed code with reentrancy protection
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract FixedContract is ReentrancyGuard {
    function withdraw(uint amount) public nonReentrant {
        require(balance[msg.sender] >= amount);
        balance[msg.sender] -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
```

### 3. Secure Code Generation

Generate secure contract templates following best practices:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract MyToken is ERC20, Ownable {
    constructor() ERC20("MyToken", "MTK") Ownable(msg.sender) {
        _mint(msg.sender, 1000000 * 10 ** decimals());
    }
}
```

### 4. Test Generation

Automatically generate comprehensive test suites:

```javascript
const { expect } = require("chai");

describe("MyToken", function () {
  let MyToken, token, owner, addr1;

  beforeEach(async function () {
    MyToken = await ethers.getContractFactory("MyToken");
    [owner, addr1] = await ethers.getSigners();
    token = await MyToken.deploy();
    await token.deployed();
  });

  it("should mint initial supply to owner", async function () {
    const balance = await token.balanceOf(owner.address);
    expect(balance).to.equal(ethers.utils.parseEther("1000000"));
  });
});
```

## 🛠️ Technical Implementation

### Model Architecture

- **Base Model**: DeepSeek-Coder-1.3B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization for memory efficiency
- **Training Optimization**: Gradient checkpointing, mixed precision training

### Training Configuration

```python
# Model Configuration
model_name: "deepseek-ai/deepseek-coder-1.3b-base"

# LoRA Configuration
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# Training Configuration
num_epochs: 3
batch_size: 4
learning_rate: 2e-4
max_seq_length: 2048
```

### Data Processing

The system processes four types of training samples:

1. **Security Audit**: Vulnerability detection and analysis
2. **Vulnerability Fix**: Code repair and optimization
3. **Code Explanation**: Function and risk explanation
4. **Best Practices**: Security guidelines and recommendations

## 📈 Performance Metrics

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8GB     | 16GB+       |
| RAM       | 16GB    | 32GB+       |
| Storage   | 20GB    | 50GB+       |

### Training Performance

- **Data Collection**: 30-60 minutes
- **Data Processing**: 5-10 minutes
- **Model Training**: 2-6 hours
- **Inference Speed**: Real-time response

### Expected Outcomes

After fine-tuning, the model can:
- Accurately identify common security vulnerabilities
- Provide specific repair recommendations
- Generate secure code examples
- Explain code functionality and risks

## 🔒 Security Considerations

### Data Security
- Only collects publicly available GitHub code
- No sensitive information included
- Complies with GitHub terms of service

### Model Security
- Intended for security research and learning only
- Not recommended for production use without additional validation
- Should be used alongside professional security tools

### Usage Guidelines
- Comply with relevant laws and regulations
- Follow ethical guidelines
- Do not use for malicious purposes

## 🧪 Testing and Validation

### Interactive Testing

```bash
# Start interactive testing
python inference.py --mode interactive
```

### Batch Testing

```bash
# Run batch tests
python inference.py --mode batch --test_file test_cases.json
```

### Creating Test Cases

```bash
# Generate test cases
python inference.py --create_test
```

## 🔧 Advanced Usage

### Custom Training

1. **Modify Data Processor**
   ```python
   # Add custom prompt templates in data_processor.py
   self.prompts['solidity']['custom_task'] = "Custom task description"
   ```

2. **Adjust LoRA Configuration**
   ```python
   # Modify LoRA parameters in train.py
   lora_config = LoraConfig(
       r=32,  # Increase rank
       lora_alpha=64,
       target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
   )
   ```

3. **Use Custom Datasets**
   ```python
   # Prepare custom data
   custom_data = [
       {"instruction": "...", "input": "", "output": "..."}
   ]
   ```

### Integration Options

- **VS Code Extension**: Language Server Protocol support
- **CLI Tool**: Command-line interface for batch processing
- **Web Interface**: Integration with Web3 development platforms
- **API Service**: RESTful API for external applications

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Insufficient**
   ```bash
   # Reduce batch size
   batch_size: int = 2
   gradient_accumulation_steps: int = 8
   ```

2. **GitHub API Rate Limiting**
   ```bash
   # Use GitHub token
   python run_finetune.py --github_token YOUR_TOKEN
   ```

3. **Dependency Installation Failures**
   ```bash
   # Use conda for PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

4. **Empty Dataset**
   ```bash
   # Check data collection success
   ls collected_contracts/solidity/
   ls vulnerability_cases/
   ```

### Log Monitoring

Training logs are saved in the `logs/` directory:

```bash
# View with TensorBoard
tensorboard --logdir logs/
```

## 📚 Documentation

- [Project Design Document](Blockchain_Coding_Agent_Design.markdown) - Comprehensive project architecture and design
- [Data Collection Guide](data/README.md) - Smart contract and vulnerability data collection
- [Fine-tuning Guide](finetune/README.md) - Model training and fine-tuning process
- [Project Summary](finetune/PROJECT_SUMMARY.md) - Detailed project overview and results

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests to improve this project.

### Development Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes only. Please comply with relevant laws and regulations.

## ⚠️ Disclaimer

- This tool is designed for security research and educational purposes
- Generated code should be thoroughly reviewed before production use
- Always follow security best practices and conduct professional audits
- The authors are not responsible for any security incidents or financial losses

## 🔗 Related Resources

- [OpenZeppelin Contracts](https://openzeppelin.com/contracts/) - Secure smart contract library
- [ConsenSys Diligence](https://consensys.net/diligence/) - Smart contract security
- [Trail of Bits](https://www.trailofbits.com/) - Security research and audits
- [DeFi Hack Labs](https://defihacklabs.com/) - DeFi security research

---

**SecureChain Agent** - Making blockchain development more secure, one contract at a time. 🔒⚡ 