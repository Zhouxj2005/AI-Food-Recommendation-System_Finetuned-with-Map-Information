import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import pandas as pd
import wandb

MAX_LENGTH = 2048
PROMPT = "你是一个美食推荐专家，请根据用户的问题给出回答。"

def process_func(example, tokenizer):
    """
    将数据集进行预处理
    """
    
    user_question = example.get("input", "")
    instruction_text = f"System: {PROMPT}\nUser: {user_question}\nAssistant: "
    instruction = tokenizer(
        instruction_text,
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 初始化 Weights & Biases
wandb.init(
    project="food-recommendation",  # 项目名称
    name="qwen3-0.6B-finetune-v3",    # 运行名称
    config={
        "model": "Qwen3-0.6B",
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
    }
)

def main():
    # 下载模型
    model_dir = snapshot_download(
        "Qwen/Qwen3-0.6B",
        cache_dir="./model_cache", # 下载到当前目录
        revision="master"
    )

    print(f"模型已下载到: {model_dir}") # 打印出来看看实际路径

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型未正确下载到 {model_dir}")

    # 加载 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    # 加载数据集
    train_df = pd.read_json("train_format.json")
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(lambda x: process_func(x, tokenizer), remove_columns=train_ds.column_names)

    eval_df = pd.read_json("eval_format.json")
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(lambda x: process_func(x, tokenizer), remove_columns=eval_ds.column_names)

    # 配置训练参数
    args = TrainingArguments(
        output_dir="./food_recommend",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=8,
        save_steps=300,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="wandb",  # 启用 wandb 可视化

        remove_unused_columns=False
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 开始训练
    trainer.train()

    # 结束 wandb 运行
    wandb.finish()

if __name__ == "__main__":
    main()
