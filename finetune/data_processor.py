#!/usr/bin/env python3
"""
区块链代码数据预处理脚本
用于处理收集到的智能合约代码，生成适合微调的数据集
"""

import os
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from prompts import (
    USER_PROMPTS,
    VULNERABILITY_AUDIT_RESPONSE,
    VULNERABILITY_FIX_RESPONSE,
    REENTRANCY_FIX_SOLIDITY_RESPONSE,
    OVERFLOW_FIX_SOLIDITY_RESPONSE,
    ACCESS_CONTROL_FIX_RESPONSE,
    GENERAL_RESPONSE,
)


class BlockchainDataProcessor:
    def __init__(
        self,
        data_dir: str = "../collected_contracts",
        output_dir: str = "processed_data",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 支持的编程语言
        self.languages = ["solidity", "rust", "vyper"]

        # 代码模板和提示词
        self.prompts = USER_PROMPTS

    def load_collected_data(self) -> Dict[str, List[Dict]]:
        """加载收集到的智能合约数据"""
        data = {}

        for lang in self.languages:
            lang_dir = self.data_dir / lang
            if not lang_dir.exists():
                print(f"警告: {lang_dir} 不存在")
                data[lang] = []
                continue

            # 加载metadata.json
            metadata_file = lang_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data[lang] = json.load(f)
                print(f"加载 {lang} 数据: {len(data[lang])} 个文件")
            else:
                print(f"警告: {metadata_file} 不存在")
                data[lang] = []

        return data

    def load_vulnerability_cases(self) -> List[Dict]:
        """加载漏洞案例数据"""
        vuln_dir = Path("../vulnerability_cases")
        cases = []

        if not vuln_dir.exists():
            print("警告: vulnerability_cases 目录不存在")
            return cases

        # 加载已知漏洞案例
        all_cases_file = vuln_dir / "all_vulnerability_cases.json"
        if all_cases_file.exists():
            with open(all_cases_file, "r", encoding="utf-8") as f:
                cases = json.load(f)
            print(f"加载漏洞案例: {len(cases)} 个")

        return cases

    def clean_code(self, code: str) -> str:
        """清理代码，移除注释和空行"""
        # 移除单行注释
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            # 跳过空行和只包含注释的行
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue

            # 移除行内注释
            if "//" in line:
                line = line.split("//")[0].rstrip()
            if "#" in line:
                line = line.split("#")[0].rstrip()

            if line.strip():
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def extract_code_from_file(self, file_path: str) -> str:
        """从文件中提取代码内容"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 移除文件头部的注释信息
            lines = content.split("\n")
            code_start = 0

            for i, line in enumerate(lines):
                if line.startswith("//") or line.startswith("#"):
                    if "=" in line and len(line) > 50:  # 分隔线
                        code_start = i + 1
                        break
                elif (
                    line.strip()
                    and not line.startswith("//")
                    and not line.startswith("#")
                ):
                    code_start = i
                    break

            return "\n".join(lines[code_start:])

        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return ""

    def generate_training_samples(
        self, data: Dict[str, List[Dict]], vuln_cases: List[Dict]
    ) -> List[Dict]:
        """生成训练样本"""
        samples = []

        # 处理常规合约代码
        for lang, contracts in data.items():
            if not contracts:
                continue

            for contract in tqdm(contracts, desc=f"处理 {lang} 合约"):
                try:
                    # 提取代码内容
                    if "content" in contract:
                        code = contract["content"]
                    else:
                        # 尝试从本地文件读取
                        repo_name = contract["repository"].split("/")[-1]
                        filename = contract["name"]
                        local_file = self.data_dir / lang / f"{repo_name}_{filename}"
                        if local_file.exists():
                            code = self.extract_code_from_file(str(local_file))
                        else:
                            continue

                    if not code or len(code.strip()) < 50:
                        continue

                    # 清理代码
                    cleaned_code = self.clean_code(code)
                    if len(cleaned_code.strip()) < 30:
                        continue

                    # 生成不同类型的训练样本
                    for prompt_type, prompt in self.prompts[lang].items():
                        # 根据合约信息生成相应的回答
                        if "vulnerability" in contract.get("query", "").lower():
                            # 这是漏洞相关的代码
                            if prompt_type == "security_audit":
                                response = self.generate_vulnerability_audit_response(
                                    cleaned_code, lang
                                )
                            elif prompt_type == "vulnerability_fix":
                                response = self.generate_vulnerability_fix_response(
                                    cleaned_code, lang
                                )
                            else:
                                response = self.generate_general_response(
                                    cleaned_code, lang, prompt_type
                                )
                        else:
                            # 这是普通代码
                            response = self.generate_general_response(
                                cleaned_code, lang, prompt_type
                            )

                        if response:
                            samples.append(
                                {
                                    "instruction": prompt + cleaned_code,
                                    "input": "",
                                    "output": response,
                                    "language": lang,
                                    "type": prompt_type,
                                    "source": contract.get("repository", "unknown"),
                                }
                            )

                except Exception as e:
                    print(f"处理合约时出错: {e}")
                    continue

        # 处理漏洞案例
        for case in tqdm(vuln_cases, desc="处理漏洞案例"):
            try:
                lang = case.get("language", "solidity")
                if lang not in self.languages:
                    continue

                # 读取案例文件
                case_files = case.get("files", [])
                for file_info in case_files:
                    local_path = file_info.get("local_path", "")
                    if local_path and os.path.exists(local_path):
                        code = self.extract_code_from_file(local_path)
                        if not code or len(code.strip()) < 30:
                            continue

                        cleaned_code = self.clean_code(code)

                        # 生成漏洞分析样本
                        for prompt_type, prompt in self.prompts[lang].items():
                            if prompt_type == "security_audit":
                                response = self.generate_vulnerability_case_response(
                                    case, cleaned_code, lang
                                )
                            elif prompt_type == "vulnerability_fix":
                                response = self.generate_vulnerability_fix_response(
                                    cleaned_code, lang, case
                                )
                            else:
                                response = self.generate_general_response(
                                    cleaned_code, lang, prompt_type
                                )

                            if response:
                                samples.append(
                                    {
                                        "instruction": prompt + cleaned_code,
                                        "input": "",
                                        "output": response,
                                        "language": lang,
                                        "type": prompt_type,
                                        "source": f"vulnerability_case_{case['name']}",
                                        "vulnerability_type": case.get(
                                            "description", "unknown"
                                        ),
                                    }
                                )

            except Exception as e:
                print(f"处理漏洞案例时出错: {e}")
                continue

        return samples

    def generate_vulnerability_audit_response(self, code: str, lang: str) -> str:
        """生成漏洞审计回答"""
        responses = VULNERABILITY_AUDIT_RESPONSE_PROMPTS

        return random.choice(responses.get(lang, responses["solidity"]))

    def generate_vulnerability_fix_response(
        self, code: str, lang: str, case: Dict = None
    ) -> str:
        """生成漏洞修复回答"""
        if case:
            # 基于具体漏洞案例生成修复代码
            vuln_type = case.get("description", "").lower()
            if "reentrancy" in vuln_type:
                return self.generate_reentrancy_fix(code, lang)
            elif "overflow" in vuln_type:
                return self.generate_overflow_fix(code, lang)
            elif "access control" in vuln_type:
                return self.generate_access_control_fix(code, lang)

        # 通用修复建议
        fixes = VULNERABILITY_FIX_RESPONSE

        return fixes.get(lang, fixes["solidity"])

    def generate_reentrancy_fix(self, code: str, lang: str) -> str:
        """生成重入攻击修复代码"""
        if lang == "solidity":
            return REENTRANCY_FIX_SOLIDITY_RESPONSE
        return "请参考上述修复方案，根据具体代码实现相应的重入保护机制。"

    def generate_overflow_fix(self, code: str, lang: str) -> str:
        """生成整数溢出修复代码"""
        if lang == "solidity":
            return OVERFLOW_FIX_SOLIDITY_RESPONSE

        return "请根据具体代码选择合适的溢出保护方案。"

    def generate_access_control_fix(self, code: str, lang: str) -> str:
        """生成访问控制修复代码"""
        if lang == "solidity":
            return ACCESS_CONTROL_FIX_RESPONSE
        return "请根据具体需求实现适当的访问控制机制。"

    def generate_general_response(self, code: str, lang: str, prompt_type: str) -> str:
        """生成通用回答"""
        responses = GENERAL_RESPONSE

        return responses.get(
            prompt_type, "这是一个智能合约代码，建议进行全面的安全审计和测试。"
        )

    def generate_vulnerability_case_response(
        self, case: Dict, code: str, lang: str
    ) -> str:
        """生成漏洞案例分析回答"""
        name = case.get("name", "Unknown")
        description = case.get("description", "")
        impact = case.get("impact", "")
        year = case.get("year", "")

        return VULNERABILITY_CASE_RESPONSE

    def create_dataset(self, samples: List[Dict]) -> DatasetDict:
        """创建HuggingFace数据集"""
        if not samples:
            print("警告: 没有生成任何训练样本")
            return DatasetDict()

        # 转换为DataFrame
        df = pd.DataFrame(samples)

        # 分割数据集
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))

        train_df = df[:train_size]
        val_df = df[train_size : train_size + val_size]
        test_df = df[train_size + val_size :]

        # 创建数据集
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return dataset_dict

    def save_dataset(self, dataset: DatasetDict, output_path: str = None):
        """保存数据集"""
        if output_path is None:
            output_path = self.output_dir / "blockchain_dataset"

        dataset.save_to_disk(output_path)
        print(f"数据集已保存到: {output_path}")

        # 保存统计信息
        stats = {
            "total_samples": sum(len(dataset[split]) for split in dataset.keys()),
            "train_samples": len(dataset["train"]),
            "validation_samples": len(dataset["validation"]),
            "test_samples": len(dataset["test"]),
            "languages": dataset["train"].unique("language"),
            "types": dataset["train"].unique("type"),
        }

        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"数据集统计信息已保存到: {stats_file}")

    def process_all(self):
        """处理所有数据"""
        print("开始处理区块链代码数据...")

        # 加载数据
        data = self.load_collected_data()
        vuln_cases = self.load_vulnerability_cases()

        # 生成训练样本
        samples = self.generate_training_samples(data, vuln_cases)

        print(f"生成了 {len(samples)} 个训练样本")

        # 创建数据集
        dataset = self.create_dataset(samples)

        # 保存数据集
        self.save_dataset(dataset)

        print("数据处理完成！")
        return dataset


def main():
    processor = BlockchainDataProcessor()
    dataset = processor.process_all()

    # 显示数据集信息
    print("\n数据集信息:")
    for split, ds in dataset.items():
        print(f"{split}: {len(ds)} 样本")

    # 显示示例
    if len(dataset["train"]) > 0:
        print("\n训练样本示例:")
        example = dataset["train"][0]
        print(f"语言: {example['language']}")
        print(f"类型: {example['type']}")
        print(f"指令: {example['instruction'][:100]}...")
        print(f"输出: {example['output'][:100]}...")


if __name__ == "__main__":
    main()
