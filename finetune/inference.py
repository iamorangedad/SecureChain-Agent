#!/usr/bin/env python3
"""
微调后的区块链代码模型推理脚本
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainCodeInference:
    def __init__(self, model_path: str, base_model: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和分词器
        self.tokenizer, self.model = self.load_model()
        
        logger.info(f"模型加载完成，使用设备: {self.device}")
    
    def load_model(self):
        """加载模型和分词器"""
        logger.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("加载基础模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        )
        
        logger.info("加载LoRA适配器...")
        model = PeftModel.from_pretrained(base_model, self.model_path)
        
        return tokenizer, model
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """格式化提示词"""
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """生成回答"""
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 生成参数
        generation_config = {
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分
        response = response[len(prompt):].strip()
        
        return response
    
    def security_audit(self, code: str, language: str = "solidity") -> str:
        """安全审计"""
        instruction = f"请对以下{language}智能合约进行安全审计，识别潜在的安全漏洞：\n\n{code}"
        return self.generate_response(instruction)
    
    def vulnerability_fix(self, code: str, language: str = "solidity") -> str:
        """漏洞修复"""
        instruction = f"请修复以下{language}智能合约中的安全漏洞：\n\n{code}"
        return self.generate_response(instruction)
    
    def code_explanation(self, code: str, language: str = "solidity") -> str:
        """代码解释"""
        instruction = f"请解释以下{language}智能合约的功能和潜在风险：\n\n{code}"
        return self.generate_response(instruction)
    
    def best_practices(self, code: str, language: str = "solidity") -> str:
        """最佳实践建议"""
        instruction = f"请分析以下{language}智能合约，并提供最佳实践建议：\n\n{code}"
        return self.generate_response(instruction)
    
    def interactive_mode(self):
        """交互模式"""
        print("欢迎使用区块链代码安全助手！")
        print("支持的功能：")
        print("1. 安全审计 (audit)")
        print("2. 漏洞修复 (fix)")
        print("3. 代码解释 (explain)")
        print("4. 最佳实践 (best)")
        print("5. 自定义指令 (custom)")
        print("输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            try:
                # 选择功能
                print("\n请选择功能 (1-5): ", end="")
                choice = input().strip()
                
                if choice.lower() == 'quit':
                    break
                
                if choice not in ['1', '2', '3', '4', '5']:
                    print("无效选择，请重新输入")
                    continue
                
                # 输入代码
                print("\n请输入代码 (输入 'END' 结束):")
                code_lines = []
                while True:
                    line = input()
                    if line.strip() == 'END':
                        break
                    code_lines.append(line)
                
                code = '\n'.join(code_lines)
                if not code.strip():
                    print("代码不能为空")
                    continue
                
                # 选择语言
                print("\n请选择语言 (solidity/rust/vyper): ", end="")
                language = input().strip().lower()
                if language not in ['solidity', 'rust', 'vyper']:
                    language = 'solidity'
                
                # 生成回答
                print("\n正在生成回答...")
                
                if choice == '1':
                    response = self.security_audit(code, language)
                elif choice == '2':
                    response = self.vulnerability_fix(code, language)
                elif choice == '3':
                    response = self.code_explanation(code, language)
                elif choice == '4':
                    response = self.best_practices(code, language)
                elif choice == '5':
                    print("请输入自定义指令: ", end="")
                    custom_instruction = input().strip()
                    instruction = f"{custom_instruction}\n\n{code}"
                    response = self.generate_response(instruction)
                
                print("\n" + "="*50)
                print("回答:")
                print(response)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\n\n退出程序")
                break
            except Exception as e:
                print(f"发生错误: {e}")
    
    def batch_test(self, test_file: str):
        """批量测试"""
        if not os.path.exists(test_file):
            print(f"测试文件不存在: {test_file}")
            return
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"测试案例 {i+1}/{len(test_cases)}: {test_case.get('name', 'Unknown')}")
            
            code = test_case.get('code', '')
            language = test_case.get('language', 'solidity')
            test_type = test_case.get('type', 'security_audit')
            
            if test_type == 'security_audit':
                response = self.security_audit(code, language)
            elif test_type == 'vulnerability_fix':
                response = self.vulnerability_fix(code, language)
            elif test_type == 'code_explanation':
                response = self.code_explanation(code, language)
            elif test_type == 'best_practices':
                response = self.best_practices(code, language)
            else:
                instruction = test_case.get('instruction', '')
                response = self.generate_response(instruction)
            
            result = {
                'test_case': test_case,
                'response': response,
                'timestamp': torch.cuda.Event() if torch.cuda.is_available() else None
            }
            
            results.append(result)
            
            print(f"完成测试案例 {i+1}")
        
        # 保存结果
        output_file = f"test_results_{len(results)}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"测试完成，结果保存到: {output_file}")

def create_test_cases():
    """创建测试案例"""
    test_cases = [
        {
            "name": "重入攻击漏洞",
            "type": "security_audit",
            "language": "solidity",
            "code": """contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0);
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success);
        
        balances[msg.sender] = 0;
    }
}"""
        },
        {
            "name": "整数溢出漏洞",
            "type": "vulnerability_fix",
            "language": "solidity",
            "code": """contract OverflowContract {
    mapping(address => uint256) public balances;
    
    function addBalance(address user, uint256 amount) public {
        balances[user] += amount;
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}"""
        },
        {
            "name": "访问控制漏洞",
            "type": "best_practices",
            "language": "solidity",
            "code": """contract AccessControlContract {
    address public owner;
    mapping(address => uint256) public balances;
    
    constructor() {
        owner = msg.sender;
    }
    
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        require(amount > 0);
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
    
    function emergencyWithdraw() public {
        payable(msg.sender).transfer(address(this).balance);
    }
}"""
        },
        {
            "name": "Rust智能合约",
            "type": "code_explanation",
            "language": "rust",
            "code": """use anchor_lang::prelude::*;

declare_id!("your_program_id");

#[program]
pub mod simple_program {
    use super::*;
    
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let state = &mut ctx.accounts.state;
        state.authority = ctx.accounts.authority.key();
        Ok(())
    }
    
    pub fn transfer(ctx: Context<Transfer>, amount: u64) -> Result<()> {
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
}"""
        }
    ]
    
    with open("test_cases.json", 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    print("测试案例已创建: test_cases.json")

def main():
    parser = argparse.ArgumentParser(description='区块链代码模型推理')
    parser.add_argument('--model_path', default='blockchain_coder_model', help='模型路径')
    parser.add_argument('--base_model', default='deepseek-ai/deepseek-coder-1.3b-base', help='基础模型')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive', help='运行模式')
    parser.add_argument('--test_file', default='test_cases.json', help='测试文件')
    parser.add_argument('--create_test', action='store_true', help='创建测试案例')
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_cases()
        return
    
    # 创建推理器
    inference = BlockchainCodeInference(args.model_path, args.base_model)
    
    if args.mode == 'interactive':
        inference.interactive_mode()
    elif args.mode == 'batch':
        inference.batch_test(args.test_file)

if __name__ == "__main__":
    main() 