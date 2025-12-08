import json
import random
import os

def split_dataset(input_file, train_file, eval_file, train_weight=8, eval_weight=1):
    """
    读取 input_file，按照 train_weight : eval_weight 的比例切分数据。
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    print(f"正在读取 {input_file} ...")
    
    # 2. 读取原始 JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("错误: JSON 文件格式不正确，请检查是否为标准的 JSON 列表格式。")
        return

    total_count = len(data)
    print(f"原始数据共 {total_count} 条。")

    if total_count == 0:
        print("数据为空，无法切分。")
        return

    # 3. 打乱数据 (非常重要！防止原本数据是排序过的导致验证集偏差)
    # 设置随机种子，保证每次运行脚本切分的结果是一样的（可复现）
    random.seed(42) 
    random.shuffle(data)

    # 4. 计算切分点
    # 计算比例: 训练集占比 = 8 / (8 + 1) = 8/9
    split_ratio = train_weight / (train_weight + eval_weight)
    split_index = int(total_count * split_ratio)

    # 5. 切分列表
    train_data = data[:split_index]
    eval_data = data[split_index:]

    # 6. 保存到文件
    print(f"正在写入 {train_file} (包含 {len(train_data)} 条数据)...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    print(f"正在写入 {eval_file} (包含 {len(eval_data)} 条数据)...")
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=4)

    print("-" * 30)
    print("✅ 数据切分完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(eval_data)} 条")

if __name__ == "__main__":
    # 配置你的文件名
    INPUT_FILE = "./data.json"          # 你的原始大文件
    TRAIN_OUTPUT = "./train.json" # 输出的训练集 (对应你 train.py 里的读取文件名)
    EVAL_OUTPUT = "./eval.json"   # 输出的验证集 (对应你 train.py 里的读取文件名)

    # 执行切分 (8:1 比例)
    split_dataset(INPUT_FILE, TRAIN_OUTPUT, EVAL_OUTPUT, train_weight=8, eval_weight=1)