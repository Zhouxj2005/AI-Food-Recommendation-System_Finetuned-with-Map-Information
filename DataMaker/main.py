import os
from agent.data_generator_agent import DataGeneratorAgent

os.environ["GAODE_KEY"] = "c81295d3a4d1b4f85ddf4bf082674de1"

agent = DataGeneratorAgent()  # 你的大模型

position = [
    "北京市北京科技大学",
    "北京市北方工业大学",
    "北京市北京化工大学",
    "北京市北京工商大学",
    "北京市北京服装学院",
    "北京市北京邮电大学",
    "北京市北京印刷学院",
    "北京市北京建筑大学",
    "北京市北京师范大学",
    "北京市中国农业大学",
    "北京市北京理工大学",
    "北京市北京大学",
    "北京市中国人民大学",
    "北京市清华大学",
    "北京市北京交通大学",
    "北京市北京工业大学",
    "北京市北京航空航天大学",
]
with open("data_raw.json", "w", encoding="utf-8") as f:
    for i in range(4):
        for pos in position:
            res = agent.generate_one(pos,nums=10)
            f.write(res + "\n")

