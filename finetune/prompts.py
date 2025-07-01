USER_PROMPTS = {
    "solidity": {
        "security_audit": "请对以下Solidity智能合约进行安全审计，识别潜在的安全漏洞：\n\n",
        "vulnerability_fix": "请修复以下Solidity智能合约中的安全漏洞：\n\n",
        "code_explanation": "请解释以下Solidity智能合约的功能和潜在风险：\n\n",
        "best_practices": "请分析以下Solidity智能合约，并提供最佳实践建议：\n\n",
    },
    "rust": {
        "security_audit": "请对以下Rust智能合约程序进行安全审计，识别潜在的安全漏洞：\n\n",
        "vulnerability_fix": "请修复以下Rust智能合约程序中的安全漏洞：\n\n",
        "code_explanation": "请解释以下Rust智能合约程序的功能和潜在风险：\n\n",
        "best_practices": "请分析以下Rust智能合约程序，并提供最佳实践建议：\n\n",
    },
    "vyper": {
        "security_audit": "请对以下Vyper智能合约进行安全审计，识别潜在的安全漏洞：\n\n",
        "vulnerability_fix": "请修复以下Vyper智能合约中的安全漏洞：\n\n",
        "code_explanation": "请解释以下Vyper智能合约的功能和潜在风险：\n\n",
        "best_practices": "请分析以下Vyper智能合约，并提供最佳实践建议：\n\n",
    },
}

VULNERABILITY_AUDIT_RESPONSE = {
    "solidity": [
        "经过安全审计，发现以下潜在漏洞：\n\n1. **重入攻击风险**: 代码中存在外部调用，可能存在重入攻击漏洞\n2. **整数溢出**: 未使用SafeMath库，存在整数溢出风险\n3. **访问控制**: 缺少适当的访问控制机制\n4. **时间戳依赖**: 使用了block.timestamp，可能被矿工操纵\n\n建议：\n- 使用ReentrancyGuard防止重入攻击\n- 使用SafeMath或Solidity 0.8+的溢出检查\n- 添加onlyOwner修饰符\n- 避免依赖block.timestamp进行关键逻辑",
        "安全审计发现多个严重漏洞：\n\n**高危漏洞**:\n- 重入攻击漏洞：外部调用前未更新状态\n- 整数溢出：未进行溢出检查\n- 访问控制缺失：关键函数缺少权限控制\n\n**中危漏洞**:\n- 时间戳依赖：使用block.timestamp\n- Gas限制：未考虑gas消耗\n\n**修复建议**:\n1. 实现Checks-Effects-Interactions模式\n2. 使用SafeMath库\n3. 添加访问控制修饰符\n4. 使用oracle替代时间戳",
    ],
    "rust": [
        "Rust智能合约安全审计结果：\n\n**发现的问题**:\n1. 签名验证不完整\n2. 缺少边界检查\n3. 错误处理不当\n4. 状态管理问题\n\n**安全建议**:\n- 实现完整的签名验证\n- 添加输入验证\n- 改进错误处理机制\n- 使用安全的随机数生成器"
    ],
    "vyper": [
        "Vyper合约安全分析：\n\n**潜在风险**:\n1. 重入攻击风险\n2. 整数溢出\n3. 访问控制问题\n\n**改进建议**:\n- 使用@nonreentrant装饰器\n- 添加访问控制\n- 验证所有输入参数"
    ],
}

VULNERABILITY_FIX_RESPONSE = {
    "solidity": """
        修复后的代码：
        ```solidity
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.0;

        import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
        import "@openzeppelin/contracts/access/Ownable.sol";
        import "@openzeppelin/contracts/utils/math/SafeMath.sol";

        contract SecureContract is ReentrancyGuard, Ownable {
            using SafeMath for uint256;
            
            mapping(address => uint256) private balances;
            
            // 添加重入保护
            function withdraw() external nonReentrant {
                uint256 amount = balances[msg.sender];
                require(amount > 0, "No balance to withdraw");
                
                // 先更新状态
                balances[msg.sender] = 0;
                
                // 最后进行外部调用
                (bool success, ) = msg.sender.call{value: amount}("");
                require(success, "Transfer failed");
            }
            
            // 添加访问控制
            function emergencyWithdraw() external onlyOwner {
                // 紧急提款逻辑
            }
        }
        ```

        主要修复：
        1. 使用ReentrancyGuard防止重入攻击
        2. 使用SafeMath进行安全数学运算
        3. 添加Ownable访问控制
        4. 实现Checks-Effects-Interactions模式""",
    "rust": """
        修复后的Rust程序：
        ```rust
        use anchor_lang::prelude::*;
        use anchor_lang::solana_program::system_program;

        declare_id!("your_program_id");

        #[program]
        pub mod secure_program {
            use super::*;
            
            pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
                // 添加适当的验证
                require!(ctx.accounts.authority.key() != Pubkey::default(), CustomError::InvalidAuthority);
                
                let state = &mut ctx.accounts.state;
                state.authority = ctx.accounts.authority.key();
                state.bump = *ctx.bumps.get("state").unwrap();
                
                Ok(())
            }
            
            pub fn secure_transfer(ctx: Context<SecureTransfer>, amount: u64) -> Result<()> {
                // 添加边界检查
                require!(amount > 0, CustomError::InvalidAmount);
                require!(ctx.accounts.from.key() == ctx.accounts.authority.key(), CustomError::Unauthorized);
                
                // 安全的转账逻辑
                let transfer_instruction = anchor_lang::solana_program::system_instruction::transfer(
                    &ctx.accounts.from.key(),
                    &ctx.accounts.to.key(),
                    amount,
                );
                
                anchor_lang::solana_program::program::invoke(
                    &transfer_instruction,
                    &[
                        ctx.accounts.from.to_account_info(),
                        ctx.accounts.to.to_account_info(),
                    ],
                )?;
                
                Ok(())
            }
        }

        #[derive(Accounts)]
        pub struct Initialize<'info> {
            #[account(init, payer = authority, space = 8 + 32 + 1)]
            pub state: Account<'info, State>,
            #[account(mut)]
            pub authority: Signer<'info>,
            pub system_program: Program<'info, System>,
        }

        #[derive(Accounts)]
        pub struct SecureTransfer<'info> {
            #[account(mut)]
            pub from: Signer<'info>,
            /// CHECK: This is safe as it's just used for transfer
            #[account(mut)]
            pub to: AccountInfo<'info>,
            pub authority: Account<'info, State>,
        }

        #[account]
        pub struct State {
            pub authority: Pubkey,
            pub bump: u8,
        }

        #[error_code]
        pub enum CustomError {
            #[msg("Invalid authority")]
            InvalidAuthority,
            #[msg("Invalid amount")]
            InvalidAmount,
            #[msg("Unauthorized")]
            Unauthorized,
        }
        ```

        主要修复：
        1. 添加输入验证和边界检查
        2. 实现适当的访问控制
        3. 使用安全的错误处理
        4. 添加自定义错误类型""",
    "vyper": """
        修复后的Vyper合约：

        ```python
        # @version ^0.3.0

        from vyper.interfaces import ERC20

        # 事件定义
        event Transfer:
            sender: indexed(address)
            receiver: indexed(address)
            value: uint256

        # 状态变量
        owner: public(address)
        balances: public(HashMap[address, uint256])
        total_supply: public(uint256)

        @external
        def __init__(_owner: address):
            self.owner = _owner
            self.total_supply = 0

        @external
        @nonreentrant('withdraw')  # 添加重入保护
        def withdraw(amount: uint256):
            assert amount > 0, "Amount must be positive"
            assert self.balances[msg.sender] >= amount, "Insufficient balance"
            
            # 先更新状态
            self.balances[msg.sender] -= amount
            self.total_supply -= amount
            
            # 最后进行外部调用
            send(msg.sender, amount)

        @external
        def transfer(_to: address, _value: uint256) -> bool:
            assert _to != empty(address), "Invalid recipient"
            assert _value > 0, "Invalid amount"
            assert self.balances[msg.sender] >= _value, "Insufficient balance"
            
            self.balances[msg.sender] -= _value
            self.balances[_to] += _value
            
            log Transfer(msg.sender, _to, _value)
            return True

        @external
        def emergency_withdraw():
            assert msg.sender == self.owner, "Only owner can call"
            # 紧急提款逻辑
            send(self.owner, self.balance)

        @view
        @external
        def balance_of(_owner: address) -> uint256:
            return self.balances[_owner]
        ```

        主要修复：
        1. 使用@nonreentrant装饰器防止重入攻击
        2. 添加适当的断言检查
        3. 实现访问控制
        4. 使用安全的状态更新模式""",
}

REENTRANCY_FIX_SOLIDITY_RESPONSE = """
    修复重入攻击漏洞：
    ```solidity
    // 原始代码存在重入攻击风险
    // 修复后的代码：

    contract SecureContract {
        mapping(address => uint256) private balances;
        bool private locked;
        
        modifier nonReentrant() {
            require(!locked, "ReentrancyGuard: reentrant call");
            locked = true;
            _;
            locked = false;
        }
        
        function withdraw() external nonReentrant {
            uint256 amount = balances[msg.sender];
            require(amount > 0, "No balance to withdraw");
            
            // 先更新状态
            balances[msg.sender] = 0;
            
            // 最后进行外部调用
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
        }
    }
    ```

    关键修复点：
    1. 添加nonReentrant修饰符
    2. 实现Checks-Effects-Interactions模式
    3. 先更新状态，再进行外部调用
"""

OVERFLOW_FIX_SOLIDITY_RESPONSE = """
    修复整数溢出漏洞：

    ```solidity
    // 使用SafeMath库
    import "@openzeppelin/contracts/utils/math/SafeMath.sol";

    contract SecureContract {
        using SafeMath for uint256;
        
        mapping(address => uint256) private balances;
        
        function addBalance(address user, uint256 amount) external {
            // 使用SafeMath防止溢出
            balances[user] = balances[user].add(amount);
        }
        
        function transfer(address to, uint256 amount) external {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            
            balances[msg.sender] = balances[msg.sender].sub(amount);
            balances[to] = balances[to].add(amount);
        }
    }

    // 或者使用Solidity 0.8+的内置溢出检查
    // pragma solidity ^0.8.0;
    // 无需SafeMath，编译器会自动检查溢出
    ```

    修复要点：
    1. 使用SafeMath库进行安全数学运算
    2. 或升级到Solidity 0.8+使用内置溢出检查
    3. 添加适当的require检查
"""

ACCESS_CONTROL_FIX_RESPONSE = """
    修复访问控制漏洞：

    ```solidity
    import "@openzeppelin/contracts/access/Ownable.sol";

    contract SecureContract is Ownable {
        mapping(address => bool) private authorized;
        
        modifier onlyAuthorized() {
            require(authorized[msg.sender] || msg.sender == owner(), "Not authorized");
            _;
        }
        
        function addAuthorized(address user) external onlyOwner {
            authorized[user] = true;
        }
        
        function removeAuthorized(address user) external onlyOwner {
            authorized[user] = false;
        }
        
        function sensitiveFunction() external onlyAuthorized {
            // 敏感操作
        }
        
        function emergencyFunction() external onlyOwner {
            // 紧急操作
        }
    }
    ```

    修复要点：
    1. 继承Ownable合约
    2. 使用onlyOwner修饰符
    3. 实现自定义访问控制
    4. 添加权限管理函数
"""

GENERAL_RESPONSE = {
    "code_explanation": f"这是一个{lang}智能合约代码。主要功能包括：\n\n1. 状态管理：管理合约状态和用户余额\n2. 转账功能：实现代币或ETH的转账\n3. 访问控制：限制特定函数的访问权限\n4. 事件记录：记录重要的操作事件\n\n潜在风险：\n- 需要仔细检查重入攻击风险\n- 确保整数运算的安全性\n- 验证所有输入参数\n- 实现适当的错误处理",
    "best_practices": f"针对这个{lang}智能合约的最佳实践建议：\n\n1. **安全性**：\n   - 使用ReentrancyGuard防止重入攻击\n   - 实现适当的访问控制\n   - 使用SafeMath或内置溢出检查\n\n2. **Gas优化**：\n   - 优化存储布局\n   - 使用批量操作减少gas消耗\n   - 避免不必要的循环\n\n3. **可维护性**：\n   - 添加详细的注释\n   - 使用事件记录重要操作\n   - 实现升级机制\n\n4. **测试**：\n   - 编写全面的单元测试\n   - 进行安全审计\n   - 使用形式化验证工具",
}

VULNERABILITY_CASE_RESPONSE = f"""
    ## {name} 漏洞分析

    **漏洞描述**: {description}

    **影响**: {impact}

    **发生年份**: {year}

    **漏洞类型**: {description}

    **代码分析**:
    这个{lang}智能合约存在严重的安全漏洞。主要问题包括：

    1. **重入攻击风险**: 代码中存在外部调用，攻击者可能通过重入攻击窃取资金
    2. **访问控制缺失**: 关键函数缺少适当的权限控制
    3. **整数溢出**: 未使用安全的数学运算库
    4. **时间戳依赖**: 使用可被操纵的时间戳

    **修复建议**:
    - 实现Checks-Effects-Interactions模式
    - 添加访问控制修饰符
    - 使用SafeMath或内置溢出检查
    - 避免依赖block.timestamp
    - 进行全面的安全审计

    **历史教训**:
    这个案例提醒我们在开发智能合约时必须高度重视安全性，任何小的疏忽都可能导致巨大的损失。
"""
