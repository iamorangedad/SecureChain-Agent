#!/usr/bin/env python3
"""
智能合约代码收集器
从GitHub收集Solidity、Rust、Vyper的智能合约代码，重点包括已知漏洞案例
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any
import argparse
from pathlib import Path

class SmartContractCollector:
    def __init__(self, github_token: str = None, output_dir: str = "collected_contracts"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # GitHub API headers
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'SmartContractCollector/1.0'
        }
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'
    
    def search_github_repos(self, query: str, language: str, max_results: int = 100) -> List[Dict]:
        """搜索GitHub仓库"""
        repos = []
        page = 1
        
        while len(repos) < max_results:
            url = "https://api.github.com/search/repositories"
            params = {
                'q': f'{query} language:{language}',
                'sort': 'stars',
                'order': 'desc',
                'page': page,
                'per_page': min(100, max_results - len(repos))
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('items'):
                    break
                
                repos.extend(data['items'])
                page += 1
                
                # GitHub API限制
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining <= 1:
                        print(f"GitHub API限制，等待60秒...")
                        time.sleep(60)
                
            except requests.exceptions.RequestException as e:
                print(f"搜索仓库时出错: {e}")
                break
        
        return repos[:max_results]
    
    def search_github_code(self, query: str, language: str, max_results: int = 100) -> List[Dict]:
        """搜索GitHub代码"""
        code_results = []
        page = 1
        
        while len(code_results) < max_results:
            url = "https://api.github.com/search/code"
            params = {
                'q': f'{query} language:{language}',
                'sort': 'indexed',
                'order': 'desc',
                'page': page,
                'per_page': min(100, max_results - len(code_results))
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('items'):
                    break
                
                code_results.extend(data['items'])
                page += 1
                
                # GitHub API限制
                if 'X-RateLimit-Remaining' in response.headers:
                    remaining = int(response.headers['X-RateLimit-Remaining'])
                    if remaining <= 1:
                        print(f"GitHub API限制，等待60秒...")
                        time.sleep(60)
                
            except requests.exceptions.RequestException as e:
                print(f"搜索代码时出错: {e}")
                break
        
        return code_results[:max_results]
    
    def get_file_content(self, repo_owner: str, repo_name: str, file_path: str) -> str:
        """获取文件内容"""
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data.get('type') == 'file':
                import base64
                content = base64.b64decode(data['content']).decode('utf-8')
                return content
            else:
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"获取文件内容时出错: {e}")
            return ""
    
    def collect_solidity_contracts(self):
        """收集Solidity智能合约"""
        print("开始收集Solidity智能合约...")
        
        # 搜索包含漏洞的Solidity合约
        vulnerability_queries = [
            "reentrancy vulnerability",
            "overflow underflow",
            "unchecked external calls",
            "delegatecall vulnerability",
            "timestamp dependency",
            "front-running vulnerability",
            "access control vulnerability",
            "gas limit vulnerability",
            "integer overflow",
            "uninitialized storage pointer"
        ]
        
        solidity_dir = self.output_dir / "solidity"
        solidity_dir.mkdir(exist_ok=True)
        
        # 收集知名漏洞案例
        known_vulnerabilities = [
            {
                "name": "DAO Hack",
                "repo": "slockit/DAO",
                "description": "The DAO reentrancy attack"
            },
            {
                "name": "Parity Multi-Sig",
                "repo": "paritytech/parity",
                "description": "Parity multi-sig wallet vulnerability"
            },
            {
                "name": "BEC Token",
                "repo": "OpenZeppelin/openzeppelin-contracts",
                "description": "BatchOverflow vulnerability"
            }
        ]
        
        collected_contracts = []
        
        # 搜索漏洞相关代码
        for query in vulnerability_queries:
            print(f"搜索: {query}")
            code_results = self.search_github_code(query, "solidity", max_results=20)
            
            for result in code_results:
                try:
                    content = self.get_file_content(
                        result['repository']['owner']['login'],
                        result['repository']['name'],
                        result['path']
                    )
                    
                    if content:
                        contract_info = {
                            'name': result['name'],
                            'repository': result['repository']['full_name'],
                            'path': result['path'],
                            'url': result['html_url'],
                            'query': query,
                            'content': content,
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        collected_contracts.append(contract_info)
                        
                        # 保存到文件
                        filename = f"{result['repository']['name']}_{result['name']}.sol"
                        filepath = solidity_dir / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"// Repository: {result['repository']['full_name']}\n")
                            f.write(f"// Path: {result['path']}\n")
                            f.write(f"// Query: {query}\n")
                            f.write(f"// URL: {result['html_url']}\n")
                            f.write(f"// Collected: {datetime.now().isoformat()}\n")
                            f.write("// " + "="*50 + "\n\n")
                            f.write(content)
                        
                        print(f"已保存: {filename}")
                        
                except Exception as e:
                    print(f"处理文件时出错: {e}")
                    continue
        
        # 保存元数据
        metadata_file = solidity_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(collected_contracts, f, indent=2, ensure_ascii=False)
        
        print(f"Solidity合约收集完成，共收集 {len(collected_contracts)} 个文件")
    
    def collect_rust_contracts(self):
        """收集Rust智能合约（主要是Solana和Substrate）"""
        print("开始收集Rust智能合约...")
        
        rust_dir = self.output_dir / "rust"
        rust_dir.mkdir(exist_ok=True)
        
        # Rust智能合约相关查询
        rust_queries = [
            "solana program vulnerability",
            "substrate pallet vulnerability",
            "anchor program vulnerability",
            "rust smart contract",
            "solana reentrancy",
            "substrate overflow",
            "anchor security",
            "solana program exploit"
        ]
        
        collected_contracts = []
        
        for query in rust_queries:
            print(f"搜索: {query}")
            code_results = self.search_github_code(query, "rust", max_results=20)
            
            for result in code_results:
                try:
                    content = self.get_file_content(
                        result['repository']['owner']['login'],
                        result['repository']['name'],
                        result['path']
                    )
                    
                    if content:
                        contract_info = {
                            'name': result['name'],
                            'repository': result['repository']['full_name'],
                            'path': result['path'],
                            'url': result['html_url'],
                            'query': query,
                            'content': content,
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        collected_contracts.append(contract_info)
                        
                        # 保存到文件
                        filename = f"{result['repository']['name']}_{result['name']}.rs"
                        filepath = rust_dir / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"// Repository: {result['repository']['full_name']}\n")
                            f.write(f"// Path: {result['path']}\n")
                            f.write(f"// Query: {query}\n")
                            f.write(f"// URL: {result['html_url']}\n")
                            f.write(f"// Collected: {datetime.now().isoformat()}\n")
                            f.write("// " + "="*50 + "\n\n")
                            f.write(content)
                        
                        print(f"已保存: {filename}")
                        
                except Exception as e:
                    print(f"处理文件时出错: {e}")
                    continue
        
        # 保存元数据
        metadata_file = rust_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(collected_contracts, f, indent=2, ensure_ascii=False)
        
        print(f"Rust合约收集完成，共收集 {len(collected_contracts)} 个文件")
    
    def collect_vyper_contracts(self):
        """收集Vyper智能合约"""
        print("开始收集Vyper智能合约...")
        
        vyper_dir = self.output_dir / "vyper"
        vyper_dir.mkdir(exist_ok=True)
        
        # Vyper智能合约相关查询
        vyper_queries = [
            "vyper smart contract",
            "vyper vulnerability",
            "vyper exploit",
            "vyper security",
            "vyper reentrancy",
            "vyper overflow"
        ]
        
        collected_contracts = []
        
        for query in vyper_queries:
            print(f"搜索: {query}")
            code_results = self.search_github_code(query, "vyper", max_results=20)
            
            for result in code_results:
                try:
                    content = self.get_file_content(
                        result['repository']['owner']['login'],
                        result['repository']['name'],
                        result['path']
                    )
                    
                    if content:
                        contract_info = {
                            'name': result['name'],
                            'repository': result['repository']['full_name'],
                            'path': result['path'],
                            'url': result['html_url'],
                            'query': query,
                            'content': content,
                            'collected_at': datetime.now().isoformat()
                        }
                        
                        collected_contracts.append(contract_info)
                        
                        # 保存到文件
                        filename = f"{result['repository']['name']}_{result['name']}.vy"
                        filepath = vyper_dir / filename
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# Repository: {result['repository']['full_name']}\n")
                            f.write(f"# Path: {result['path']}\n")
                            f.write(f"# Query: {query}\n")
                            f.write(f"# URL: {result['html_url']}\n")
                            f.write(f"# Collected: {datetime.now().isoformat()}\n")
                            f.write("# " + "="*50 + "\n\n")
                            f.write(content)
                        
                        print(f"已保存: {filename}")
                        
                except Exception as e:
                    print(f"处理文件时出错: {e}")
                    continue
        
        # 保存元数据
        metadata_file = vyper_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(collected_contracts, f, indent=2, ensure_ascii=False)
        
        print(f"Vyper合约收集完成，共收集 {len(collected_contracts)} 个文件")
    
    def collect_all(self):
        """收集所有类型的智能合约"""
        print("开始收集所有智能合约代码...")
        
        self.collect_solidity_contracts()
        self.collect_rust_contracts()
        self.collect_vyper_contracts()
        
        print("所有智能合约代码收集完成！")
        
        # 生成总结报告
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """生成收集总结报告"""
        report_file = self.output_dir / "collection_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 智能合约代码收集报告\n\n")
            f.write(f"收集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 统计各语言文件数量
            for lang in ['solidity', 'rust', 'vyper']:
                lang_dir = self.output_dir / lang
                if lang_dir.exists():
                    metadata_file = lang_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as mf:
                            metadata = json.load(mf)
                            f.write(f"## {lang.capitalize()} 合约\n")
                            f.write(f"- 收集文件数: {len(metadata)}\n")
                            f.write(f"- 目录: {lang_dir}\n\n")
                            
                            # 列出前10个文件
                            f.write("### 收集的文件:\n")
                            for i, contract in enumerate(metadata[:10]):
                                f.write(f"{i+1}. [{contract['name']}]({contract['url']}) - {contract['repository']}\n")
                            if len(metadata) > 10:
                                f.write(f"... 还有 {len(metadata) - 10} 个文件\n")
                            f.write("\n")
            
            f.write("## 使用说明\n\n")
            f.write("1. 每个语言类型都有独立的目录\n")
            f.write("2. 每个文件都包含原始代码和元数据信息\n")
            f.write("3. metadata.json 文件包含所有收集文件的详细信息\n")
            f.write("4. 重点关注包含漏洞关键词的代码\n\n")
            
            f.write("## 注意事项\n\n")
            f.write("- 这些代码可能包含安全漏洞，仅用于学习和研究\n")
            f.write("- 请勿在生产环境中使用这些代码\n")
            f.write("- 建议结合安全审计工具进行分析\n")
        
        print(f"总结报告已生成: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='从GitHub收集智能合约代码')
    parser.add_argument('--token', help='GitHub API token (可选，用于提高API限制)')
    parser.add_argument('--output', default='collected_contracts', help='输出目录')
    parser.add_argument('--language', choices=['solidity', 'rust', 'vyper', 'all'], 
                       default='all', help='要收集的语言类型')
    
    args = parser.parse_args()
    
    collector = SmartContractCollector(github_token=args.token, output_dir=args.output)
    
    if args.language == 'all':
        collector.collect_all()
    elif args.language == 'solidity':
        collector.collect_solidity_contracts()
    elif args.language == 'rust':
        collector.collect_rust_contracts()
    elif args.language == 'vyper':
        collector.collect_vyper_contracts()

if __name__ == "__main__":
    main() 