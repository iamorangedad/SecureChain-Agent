#!/usr/bin/env python3
"""
DeepSeek-Coder-1.3B 区块链代码微调训练脚本
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_from_disk, Dataset
import wandb
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"
    output_dir: str = "blockchain_coder_model"
    
    # 数据配置
    dataset_path: str = "processed_data/blockchain_dataset"
    
    # 训练配置
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 其他配置
    seed: int = 42
    fp16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # 早停配置
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

class BlockchainCodeTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 初始化wandb
        self.init_wandb()
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型: {config.model_name}")
    
    def init_wandb(self):
        """初始化wandb"""
        try:
            wandb.init(
                project="blockchain-coder-finetune",
                name=f"deepseek-coder-1.3b-blockchain",
                config=vars(self.config),
                tags=["deepseek-coder", "blockchain", "finetune"]
            )
        except Exception as e:
            logger.warning(f"wandb初始化失败: {e}")
    
    def load_tokenizer(self):
        """加载分词器"""
        logger.info("加载分词器...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def load_model(self):
        """加载模型"""
        logger.info("加载模型...")
        
        # 使用4bit量化加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
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
        
        # 准备模型进行训练
        model = prepare_model_for_kbit_training(model)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        
        # 启用梯度检查点
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # 打印可训练参数
        model.print_trainable_parameters()
        
        return model
    
    def load_dataset(self):
        """加载数据集"""
        logger.info("加载数据集...")
        
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
        
        dataset = load_from_disk(str(dataset_path))
        logger.info(f"数据集加载完成: {dataset}")
        
        return dataset
    
    def format_prompt(self, instruction: str, input_text: str, output: str) -> str:
        """格式化提示词"""
        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
    
    def tokenize_function(self, examples):
        """分词函数"""
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = self.format_prompt(
                examples['instruction'][i],
                examples['input'][i],
                examples['output'][i]
            )
            prompts.append(prompt)
        
        # 分词
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # 设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset, tokenizer):
        """准备数据集"""
        logger.info("准备数据集...")
        
        self.tokenizer = tokenizer
        
        # 对数据集进行分词
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="分词处理"
        )
        
        return tokenized_dataset
    
    def create_data_collator(self, tokenizer):
        """创建数据整理器"""
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    
    def create_training_arguments(self):
        """创建训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if wandb.run else None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            group_by_length=True,
            length_column_name="length",
            ddp_find_unused_parameters=False,
        )
    
    def create_callbacks(self):
        """创建回调函数"""
        callbacks = []
        
        # 早停回调
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        return callbacks
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 加载组件
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        dataset = self.load_dataset()
        
        # 准备数据集
        tokenized_dataset = self.prepare_dataset(dataset, tokenizer)
        
        # 创建训练组件
        training_args = self.create_training_arguments()
        data_collator = self.create_data_collator(tokenizer)
        callbacks = self.create_callbacks()
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存最终模型
        logger.info("保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)
        
        # 保存训练配置
        config_save_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(vars(self.config), f, indent=2, ensure_ascii=False)
        
        # 评估模型
        logger.info("评估模型...")
        eval_results = trainer.evaluate()
        logger.info(f"评估结果: {eval_results}")
        
        # 保存评估结果
        eval_save_path = Path(self.config.output_dir) / "eval_results.json"
        with open(eval_save_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        logger.info("训练完成！")
        
        return trainer, eval_results

def main():
    """主函数"""
    # 创建配置
    config = TrainingConfig()
    
    # 创建训练器
    trainer = BlockchainCodeTrainer(config)
    
    # 开始训练
    trainer_instance, eval_results = trainer.train()
    
    print("训练完成！")
    print(f"模型保存在: {config.output_dir}")
    print(f"评估结果: {eval_results}")

if __name__ == "__main__":
    main() 