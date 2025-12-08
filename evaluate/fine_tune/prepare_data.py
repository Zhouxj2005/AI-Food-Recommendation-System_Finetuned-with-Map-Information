import json
from typing import List, Dict

def prepare_data():
    """
    读取 ./train.json 和 ./eval.json，并将格式化后的数据写入
    ./train_format.json 和 ./eval_format.json。
    """
    instruction_text = "你是一个美食推荐专家，请根据用户的问题给出回答。"

    def _load_and_convert(input_path: str, output_path: str):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{input_path} must be a JSON array")

        formatted_data: List[Dict[str, str]] = []
        for item in data:
            question = item.get("Question")
            answer = item.get("Answer")
            if question and answer:
                formatted_data.append({
                    "instruction": instruction_text,
                    "input": question,
                    "output": answer
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    # Process train.json
    _load_and_convert("./train.json", "./train_format.json")
    # Process eval.json
    _load_and_convert("./eval.json", "./eval_format.json")


if __name__ == "__main__":
    prepare_data()

