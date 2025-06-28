#!/usr/bin/env python3
"""
完整的区块链代码微调流程脚本
整合数据收集、处理和训练
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any

class BlockchainFinetunePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.finetune_dir = Path(__file__).parent
        
        # 创建必要的目录
        self.create_directories()
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            "collected_contracts",
            "vulnerability_cases", 
            "processed_data",
            "blockchain_coder_model",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"创建目录: {dir_path}")
    
    def run_data_collection(self):
        """运行数据收集"""
        print("=" * 60)
        print("步骤 1: 收集智能合约数据")
        print("=" * 60)
        
        # 收集通用合约数据
        print("\n1.1 收集通用智能合约数据...")
        cmd = [
            sys.executable, 
            str(self.project_root / "smart_contract_collector.py"),
            "--output", "collected_contracts"
        ]
        
        if self.config.get("github_token"):
            cmd.extend(["--token", self.config["github_token"]])
        
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✓ 通用合约数据收集完成")
        except subprocess.CalledProcessError as e:
            print(f"✗ 通用合约数据收集失败: {e}")
            return False
        
        # 收集漏洞案例数据
        print("\n1.2 收集漏洞案例数据...")
        cmd = [
            sys.executable,
            str(self.project_root / "vulnerability_collector.py"),
            "--output", "vulnerability_cases"
        ]
        
        if self.config.get("github_token"):
            cmd.extend(["--token", self.config["github_token"]])
        
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✓ 漏洞案例数据收集完成")
        except subprocess.CalledProcessError as e:
            print(f"✗ 漏洞案例数据收集失败: {e}")
            return False
        
        return True
    
    def run_data_processing(self):
        """运行数据处理"""
        print("\n" + "=" * 60)
        print("步骤 2: 处理数据")
        print("=" * 60)
        
        cmd = [
            sys.executable,
            str(self.finetune_dir / "data_processor.py")
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.finetune_dir)
            print("✓ 数据处理完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 数据处理失败: {e}")
            return False
    
    def run_training(self):
        """运行训练"""
        print("\n" + "=" * 60)
        print("步骤 3: 开始微调训练")
        print("=" * 60)
        
        # 检查数据集是否存在
        dataset_path = self.finetune_dir / "processed_data" / "blockchain_dataset"
        if not dataset_path.exists():
            print(f"✗ 数据集不存在: {dataset_path}")
            print("请先运行数据收集和处理步骤")
            return False
        
        # 检查CUDA可用性
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print("⚠ CUDA不可用，将使用CPU训练（速度较慢）")
        except ImportError:
            print("⚠ PyTorch未安装，请先安装依赖")
            return False
        
        # 运行训练
        cmd = [
            sys.executable,
            str(self.finetune_dir / "train.py")
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.finetune_dir)
            print("✓ 训练完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 训练失败: {e}")
            return False
    
    def run_evaluation(self):
        """运行评估"""
        print("\n" + "=" * 60)
        print("步骤 4: 模型评估")
        print("=" * 60)
        
        # 创建测试案例
        print("\n4.1 创建测试案例...")
        cmd = [
            sys.executable,
            str(self.finetune_dir / "inference.py"),
            "--create_test"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.finetune_dir)
            print("✓ 测试案例创建完成")
        except subprocess.CalledProcessError as e:
            print(f"✗ 测试案例创建失败: {e}")
            return False
        
        # 运行批量测试
        print("\n4.2 运行批量测试...")
        cmd = [
            sys.executable,
            str(self.finetune_dir / "inference.py"),
            "--mode", "batch",
            "--test_file", "test_cases.json"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.finetune_dir)
            print("✓ 批量测试完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 批量测试失败: {e}")
            return False
    
    def run_interactive_test(self):
        """运行交互式测试"""
        print("\n" + "=" * 60)
        print("步骤 5: 交互式测试")
        print("=" * 60)
        
        print("启动交互式测试模式...")
        print("您可以输入智能合约代码进行测试")
        print("输入 'quit' 退出测试")
        
        cmd = [
            sys.executable,
            str(self.finetune_dir / "inference.py"),
            "--mode", "interactive"
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=self.finetune_dir)
            print("✓ 交互式测试完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 交互式测试失败: {e}")
            return False
    
    def check_dependencies(self):
        """检查依赖"""
        print("=" * 60)
        print("检查依赖")
        print("=" * 60)
        
        required_packages = [
            "torch", "transformers", "datasets", "accelerate", 
            "peft", "bitsandbytes", "trl", "wandb", "tensorboard"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                print(f"✗ {package} (缺失)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n缺失的包: {', '.join(missing_packages)}")
            print("请运行以下命令安装依赖:")
            print(f"pip install -r {self.finetune_dir / 'requirements.txt'}")
            return False
        
        print("\n✓ 所有依赖已安装")
        return True
    
    def generate_report(self):
        """生成报告"""
        print("\n" + "=" * 60)
        print("生成微调报告")
        print("=" * 60)
        
        report = {
            "project": "DeepSeek-Coder-1.3B 区块链代码微调",
            "config": self.config,
            "directories": {
                "collected_contracts": str(self.project_root / "collected_contracts"),
                "vulnerability_cases": str(self.project_root / "vulnerability_cases"),
                "processed_data": str(self.finetune_dir / "processed_data"),
                "model": str(self.finetune_dir / "blockchain_coder_model")
            }
        }
        
        # 检查文件统计
        try:
            # 检查收集的数据
            solidity_dir = self.project_root / "collected_contracts" / "solidity"
            if solidity_dir.exists():
                solidity_files = len(list(solidity_dir.glob("*.sol")))
                report["data_stats"] = {"solidity_files": solidity_files}
            
            # 检查处理后的数据
            dataset_path = self.finetune_dir / "processed_data" / "blockchain_dataset"
            if dataset_path.exists():
                from datasets import load_from_disk
                dataset = load_from_disk(str(dataset_path))
                report["dataset_stats"] = {
                    "train": len(dataset["train"]),
                    "validation": len(dataset["validation"]),
                    "test": len(dataset["test"])
                }
            
            # 检查模型
            model_path = self.finetune_dir / "blockchain_coder_model"
            if model_path.exists():
                report["model_status"] = "已训练"
                
                # 读取训练配置
                config_file = model_path / "training_config.json"
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        report["training_config"] = json.load(f)
                
                # 读取评估结果
                eval_file = model_path / "eval_results.json"
                if eval_file.exists():
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        report["eval_results"] = json.load(f)
            else:
                report["model_status"] = "未训练"
        
        except Exception as e:
            print(f"生成报告时出错: {e}")
        
        # 保存报告
        report_file = self.finetune_dir / "finetune_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 报告已保存到: {report_file}")
        
        # 打印摘要
        print("\n微调项目摘要:")
        print(f"- 项目根目录: {self.project_root}")
        print(f"- 微调目录: {self.finetune_dir}")
        print(f"- 模型状态: {report.get('model_status', '未知')}")
        
        if "dataset_stats" in report:
            stats = report["dataset_stats"]
            print(f"- 训练样本: {stats['train']}")
            print(f"- 验证样本: {stats['validation']}")
            print(f"- 测试样本: {stats['test']}")
        
        if "eval_results" in report:
            eval_results = report["eval_results"]
            print(f"- 最终损失: {eval_results.get('eval_loss', 'N/A')}")
    
    def run_pipeline(self, steps: list):
        """运行完整的微调流程"""
        print("DeepSeek-Coder-1.3B 区块链代码微调流程")
        print("=" * 60)
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 运行指定步骤
        step_functions = {
            "collect": self.run_data_collection,
            "process": self.run_data_processing,
            "train": self.run_training,
            "eval": self.run_evaluation,
            "test": self.run_interactive_test
        }
        
        for step in steps:
            if step in step_functions:
                print(f"\n运行步骤: {step}")
                if not step_functions[step]():
                    print(f"步骤 {step} 失败，停止流程")
                    return False
            else:
                print(f"未知步骤: {step}")
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("微调流程完成！")
        print("=" * 60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='DeepSeek-Coder-1.3B 区块链代码微调流程')
    parser.add_argument('--steps', nargs='+', 
                       choices=['collect', 'process', 'train', 'eval', 'test'],
                       default=['collect', 'process', 'train', 'eval'],
                       help='要运行的步骤')
    parser.add_argument('--github_token', help='GitHub API token')
    parser.add_argument('--config', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # 命令行参数覆盖配置文件
    if args.github_token:
        config['github_token'] = args.github_token
    
    # 创建管道并运行
    pipeline = BlockchainFinetunePipeline(config)
    success = pipeline.run_pipeline(args.steps)
    
    if success:
        print("\n🎉 微调流程成功完成！")
        print("\n下一步:")
        print("1. 查看生成的报告: finetune/finetune_report.json")
        print("2. 测试模型: python finetune/inference.py --mode interactive")
        print("3. 使用模型进行区块链代码安全分析")
    else:
        print("\n❌ 微调流程失败，请检查错误信息")

if __name__ == "__main__":
    main() 