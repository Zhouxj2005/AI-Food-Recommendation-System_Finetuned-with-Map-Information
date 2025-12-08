import matplotlib.pyplot as plt
import numpy as np
import json

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ================= 数据准备部分 (保持你的代码不变) =================
scores = list(range(1, 11))  # 评分范围：1-10分

def get_distribution(file_path):
    counts = np.zeros(11, dtype=int)
    # 实际运行时请确保文件路径正确，这里假设文件存在
    try:
        items = json.load(open(file_path, "r", encoding="utf-8"))
        for item in items:
            score = item.get("evaluate_score",0)
            if score == 0:
                score = item.get("composite_score", 0)
            if 1 <= score <= 10:
                counts[score - 1] += 1
    except FileNotFoundError:
        print(f"警告: 未找到文件 {file_path}，将使用全0数据代替以防止报错。")
    total = counts.sum()
    print("加载文件:", file_path, "评分分布:", counts, "总数:", total)
    percentages = np.round((counts / total) * 100, 1) if total > 0 else np.zeros(11)
    return counts, percentages

# 加载数据 (切记：绘图时我们需要把数组切片 [:10] 以匹配 1-10 分)
dataset_counts, dataset_percentages = get_distribution("./dataset_evaluation/evaluation_results.json")
# print("训练数据质量分布：", dataset_counts)

raw_counts, raw_percentages = get_distribution("./raw_mode_evaluation/evaluation_results.json")
finetuned_counts, finetuned_percentages = get_distribution("./fine-tuned_model_evaluation/evaluation_results.json")

# ================= 绘图函数封装 =================

def draw_single_bar_chart(data, title, color_hex):
    """
    绘制单张柱状图的辅助函数
    """
    plt.figure(figsize=(8, 6))  # 每次调用都创建一个新的独立画布
    
    # 截取前10个数据对应1-10分
    valid_data = data[:10]
    
    bars = plt.bar(scores, valid_data, color=color_hex, alpha=0.7, width=0.6)
    
    plt.title(title, fontsize=16)
    plt.xlabel('评分 (1-10)', fontsize=12)
    plt.ylabel('占比 (%)', fontsize=12)
    plt.xticks(scores)
    plt.ylim(0, 100) # 统一Y轴高度方便对比
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, 
                 f'{height}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show() # 显示当前窗口

# ================= 开始绘图 =================

# 1. 训练数据分布图
draw_single_bar_chart(dataset_percentages, '训练数据质量分布', '#1f77b4') # 蓝色

# 2. 原始模型分布图
draw_single_bar_chart(raw_percentages, '原始模型表现分布', '#ff7f0e') # 橙色

# 3. 微调模型分布图
draw_single_bar_chart(finetuned_percentages, '微调模型表现分布', '#2ca02c') # 绿色

# 4. 三者对比雷达图
plt.figure(figsize=(8, 8)) # 创建第4个独立画布
ax = plt.subplot(111, polar=True) # 建立极坐标系

# 数据准备
labels = ['训练数据', '原始模型', '微调模型']
data_list = [dataset_percentages[:10], raw_percentages[:10], finetuned_percentages[:10]]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 角度设置 (10个分数对应10个角度)
angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False)
# 闭环操作：把角度数组的第一个值加到最后
angles = np.concatenate((angles, [angles[0]]))

for i, data in enumerate(data_list):
    # 数据闭环：把数据的第一个值加到最后
    data_closed = np.concatenate((data, [data[0]]))
    
    ax.plot(angles, data_closed, 'o-', linewidth=2, label=labels[i], color=colors[i])
    ax.fill(angles, data_closed, alpha=0.15, color=colors[i])

# 设置雷达图属性
ax.set_thetagrids(angles[:-1] * 180 / np.pi, scores) # 设置刻度标签为 1-10
ax.set_title('三种分布雷达图对比（百分比）', fontsize=16, y=1.05)
ax.set_ylim(0, 50) # 范围0-100
ax.grid(True)
plt.legend(loc='best') # 图例

plt.tight_layout()
plt.show()