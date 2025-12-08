from openai import OpenAI
import os
import json

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key = "sk-46840bdaeb31444a80dce5444af61633",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_raw_prediction(question):
    messages = [{"role": "user", "content": question}]
    completion = client.chat.completions.create(
        model="qwen3-0.6b",  # 您可以按需更换为其它深度思考模型
        messages=messages,
        extra_body={"enable_thinking": False},
        stream=False
    )
    return completion.choices[0].message.content

predictions = []
# data = []
if __name__ == "__main__":
    with open("data.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    for i, item in enumerate(items):
        question = item.get("Question", "")
        if not question:
            print(f"第 {i} 条数据缺少 'question' 字段，跳过。")
            break
        predictions.append(get_raw_prediction(question))
        if (i + 1) % 20 == 0:
            print(f"已处理 {i + 1} 条数据。")

    with open("raw_model_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    # with open("data.json", "r", encoding="utf-8") as f:
    #     items = json.load(f)
    # for i, item in enumerate(items):
    #     instruction = item.get("Instruction", "")
    #     question = item.get("Question", "")
    #     data.append({
    #         "Instruction": instruction,
    #         "Question": question,
    #     })

    # with open("raw_model_predictions.json", "r", encoding="utf-8") as f:
    #     items = json.load(f)
    # for i, item in enumerate(items):
    #     data[i]["Answer"] = item

    # with open("raw_model_dataset.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)