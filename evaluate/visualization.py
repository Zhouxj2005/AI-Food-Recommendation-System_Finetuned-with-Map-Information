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
            score = item.get("composite_score", 0)
            if 1 <= score <= 10:
                counts[score - 1] += 1
    except FileNotFoundError:
        print(f"警告: 未找到文件 {file_path}，将使用全0数据代替以防止报错。")
    
    total = counts.sum()
    percentages = np.round((counts / total) * 100, 1) if total > 0 else np.zeros(11)
    return counts, percentages

# 加载数据 (切记：绘图时我们需要把数组切片 [:10] 以匹配 1-10 分)
dataset_counts, dataset_percentages = get_distribution("./dataset_evaluation/evaluation_results.json")
raw_counts, raw_percentages = get_distribution("./raw_mode_evaluation/evaluation_results.json")
finetuned_counts, finetuned_percentages = get_distribution("./fine-tuned_model_evaluation/evaluation_results.json")

# ================= 绘图部分 (新增代码) =================

# 创建画布，2行2列，figsize控制整体大小
fig = plt.figure(figsize=(18, 14))
fig.suptitle('模型评分分布对比分析', fontsize=20, y=0.96)

# 定义颜色，方便统一管理
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿
labels = ['训练数据质量', '原始模型表现', '微调模型表现']
data_list = [dataset_percentages[:10], raw_percentages[:10], finetuned_percentages[:10]]

# --- 绘制前三张：柱状图 ---
# 使用循环来绘制前三个子图
for i in range(3):
    ax = fig.add_subplot(2, 2, i + 1) # 位置：1, 2, 3
    data = data_list[i]
    
    # 画柱状图
    bars = ax.bar(scores, data, color=colors[i], alpha=0.7, width=0.6)
    
    # 设置标题和标签
    ax.set_title(f'{labels[i]}分布', fontsize=14)
    ax.set_xlabel('评分 (1-10)', fontsize=12)
    ax.set_ylabel('占比 (%)', fontsize=12)
    ax.set_xticks(scores) # 强制显示1-10的所有刻度
    ax.set_ylim(0, 100) # 固定Y轴范围0-100，方便直观对比
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 在柱子上显示具体数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, 
                f'{height}%', ha='center', va='bottom', fontsize=10)

# --- 绘制第四张：雷达图 (对比图) ---
# 第4个位置使用极坐标 projection='polar'
ax_radar = fig.add_subplot(2, 2, 4, polar=True)

# 雷达图由于是闭环，需要将数据的第一个点追加到最后
angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False)
angles = np.concatenate((angles, [angles[0]])) # 闭环角度

# 画三个分布的线条
for i, data in enumerate(data_list):
    # 数据也要闭环，把第一个数据加到最后
    data_closed = np.concatenate((data, [data[0]]))
    
    ax_radar.plot(angles, data_closed, 'o-', linewidth=2, label=labels[i], color=colors[i])
    ax_radar.fill(angles, data_closed, alpha=0.15, color=colors[i]) # 填充颜色

# 设置雷达图的标签 (显示在圆周上)
ax_radar.set_thetagrids(angles[:-1] * 180 / np.pi, scores)
ax_radar.set_title('三种分布雷达图对比', fontsize=14, y=1.08)
ax_radar.set_ylim(0, 100) # 这里的上限可以根据实际数据的最大值调整，比如 max(所有数据)+10
ax_radar.grid(True)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 调整布局防止重叠
plt.tight_layout()
plt.subplots_adjust(top=0.90) # 给总标题留出空间

# 保存图片或显示
# plt.savefig('score_distribution_analysis.png', dpi=300) 
plt.show()