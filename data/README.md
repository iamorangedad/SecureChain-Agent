# 智能合约代码收集工具

这个项目包含两个主要的Python脚本，用于从GitHub收集Solidity、Rust、Vyper的智能合约代码，特别关注已知漏洞案例。

## 文件说明

- `smart_contract_collector.py` - 通用智能合约代码收集器
- `vulnerability_collector.py` - 专门收集已知漏洞案例的工具
- `requirements.txt` - Python依赖包
- `README.md` - 使用说明

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 通用智能合约收集器

收集所有类型的智能合约代码：

```bash
python smart_contract_collector.py
```

收集特定语言的合约：

```bash
# 只收集Solidity合约
python smart_contract_collector.py --language solidity

# 只收集Rust合约
python smart_contract_collector.py --language rust

# 只收集Vyper合约
python smart_contract_collector.py --language vyper
```

指定输出目录：

```bash
python smart_contract_collector.py --output my_contracts
```

使用GitHub API token（提高API限制）：

```bash
python smart_contract_collector.py --token YOUR_GITHUB_TOKEN
```

### 2. 漏洞案例收集器

收集所有漏洞相关代码：

```bash
python vulnerability_collector.py
```

只收集已知漏洞案例：

```bash
python vulnerability_collector.py --type cases
```

只收集漏洞模式：

```bash
python vulnerability_collector.py --type patterns
```

## 输出结构

### 通用收集器输出

```
collected_contracts/
├── solidity/
│   ├── metadata.json
│   ├── repo1_contract1.sol
│   ├── repo2_contract2.sol
│   └── ...
├── rust/
│   ├── metadata.json
│   ├── repo1_program1.rs
│   └── ...
├── vyper/
│   ├── metadata.json
│   ├── repo1_contract1.vy
│   └── ...
└── collection_report.md
```

### 漏洞收集器输出

```
vulnerability_cases/
├── solidity/
│   ├── DAO_Hack_Reentrancy_Attack/
│   │   ├── DAO.sol
│   │   ├── TokenCreation.sol
│   │   └── case_info.json
│   ├── Parity_Multi_Sig_Wallet/
│   └── ...
├── rust/
│   ├── Wormhole_Bridge/
│   └── ...
├── vulnerability_patterns/
│   ├── solidity/
│   │   ├── reentrancy/
│   │   ├── integer_overflow/
│   │   └── ...
│   └── all_patterns.json
├── all_vulnerability_cases.json
└── vulnerability_report.md
```

## 收集的漏洞类型

### Solidity 漏洞

1. **重入攻击 (Reentrancy)**
   - DAO Hack (2016)
   - SpankChain (2018)

2. **整数溢出 (Integer Overflow)**
   - BEC Token (2018)
   - King of the Ether Throne (2016)

3. **访问控制漏洞 (Access Control)**
   - Parity Multi-Sig (2017)
   - Poly Network (2021)

4. **时间戳依赖 (Timestamp Dependency)**
   - GovernMental (2015)

5. **闪电贷攻击 (Flash Loan Attack)**
   - bZx Protocol (2020)
   - Harvest Finance (2020)
   - Cream Finance (2021)

6. **预言机操纵 (Oracle Manipulation)**
   - Mango Markets (2022)

7. **代理合约漏洞 (Proxy Contract)**
   - Wintermute (2022)

### Rust 漏洞

1. **签名验证漏洞**
   - Wormhole Bridge (2022)

2. **预言机操纵**
   - Mango Markets (2022)

## 安全提醒

⚠️ **重要警告**

- 这些代码仅用于安全研究和学习目的
- 请勿在生产环境中使用或部署这些代码
- 建议在隔离环境中进行分析
- 遵守相关法律法规和道德准则
- 这些代码可能包含严重的安全漏洞

## GitHub API 限制

- 未认证用户：每小时60次请求
- 认证用户：每小时5000次请求
- 建议使用GitHub Personal Access Token以提高限制

### 获取GitHub Token

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token"
3. 选择 "repo" 权限
4. 复制生成的token

## 故障排除

### 常见问题

1. **API限制错误**
   - 使用GitHub token
   - 减少搜索频率
   - 等待限制重置

2. **文件下载失败**
   - 检查网络连接
   - 验证仓库和文件路径
   - 确认文件存在

3. **权限错误**
   - 检查GitHub token权限
   - 确认仓库访问权限

## 贡献

欢迎提交Issue和Pull Request来改进这个工具。

## 许可证

本项目仅用于教育和研究目的。请遵守相关法律法规。 