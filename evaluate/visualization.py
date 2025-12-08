import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据准备
scores = list(range(1, 11))

# 训练数据质量分布
train_counts = [0, 0, 0, 29, 33, 156, 133, 541, 198, 0]
train_percentages = [0.0, 0.0, 0.0, 2.7, 3.0, 14.3, 12.2, 49.6, 18.2, 0.0]

# 原始模型表现分布
model_counts = [0, 0, 29, 32, 153, 134, 505, 192, 45, 0]
model_percentages = [0.0, 0.0, 2.7, 2.9, 14.0, 12.3, 46.3, 17.6, 4.1, 0.0]

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('餐厅推荐数据集质量评估报告可视化分析', fontsize=16, fontweight='bold')

# 1. 训练数据质量评分分布 (柱状图)
bars1 = ax1.bar(scores, train_counts, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('训练数据质量评分分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('评分', fontsize=12)
ax1.set_ylabel('样本数量', fontsize=12)
ax1.set_xticks(scores)
ax1.grid(axis='y', alpha=0.3)

# 在柱子上添加百分比标签
for i, (bar, perc) in enumerate(zip(bars1, train_percentages)):
    if perc > 0:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{perc}%', ha='center', va='bottom', fontsize=9)

# 2. 原始模型表现评分分布 (柱状图)
bars2 = ax2.bar(scores, model_counts, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_title('原始模型表现评分分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('评分', fontsize=12)
ax2.set_ylabel('样本数量', fontsize=12)
ax2.set_xticks(scores)
ax2.grid(axis='y', alpha=0.3)

# 在柱子上添加百分比标签
for i, (bar, perc) in enumerate(zip(bars2, model_percentages)):
    if perc > 0:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{perc}%', ha='center', va='bottom', fontsize=9)

# 3. 对比雷达图
ax3 = plt.subplot(2, 2, 3, projection='polar')

# 只选择有数据的分数（4-9分）
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
train_radar = train_percentages[3:9]
model_radar = model_percentages[3:9]

# 闭合数据
train_radar = np.append(train_radar, train_radar[0])
model_radar = np.append(model_radar, model_radar[0])
angles = np.append(angles, angles[0])

ax3.plot(angles, train_radar, 'o-', linewidth=2, label='训练数据质量', color='blue')
ax3.fill(angles, train_radar, alpha=0.25, color='blue')
ax3.plot(angles, model_radar, 'o-', linewidth=2, label='模型表现', color='red')
ax3.fill(angles, model_radar, alpha=0.25, color='red')

ax3.set_title('训练数据与模型表现对比 (4-9分)', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(['4分', '5分', '6分', '7分', '8分', '9分'])
ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax3.grid(True)

# 4. 关键指标展示
ax4.axis('off')
metrics_text = f"""
关键指标汇总

数据集信息:
• 样本总数: 1090
• 评估时间: 2025-12-07 20:33:31

训练数据质量:
• 平均分: 7.58
• 最低分: 4
• 最高分: 9
• 主要分布: 8分 (49.6%)

原始模型表现:
• 平均分: 6.66
• 最低分: 3
• 最高分: 9
• 主要分布: 7分 (46.3%)

相关性分析:
• 训练数据与模型表现相关性: 0.950

结论:
• 训练数据质量较高 (平均7.58分)
• 模型表现中等 (平均6.66分)
• 两者高度相关 (0.950)
"""

ax4.text(0.1, 0.5, metrics_text, fontsize=12,
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 保存图表
plt.savefig('evaluation_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印数据统计
print("=" * 60)
print("数据统计摘要")
print("=" * 60)
print(f"训练数据质量平均分: {7.58}")
print(f"原始模型表现平均分: {6.66}")
print(f"两者相关性: {0.950}")
print(f"训练数据8-9分占比: {(541+198)/1090*100:.1f}%")
print(f"模型表现7-8分占比: {(505+192)/1090*100:.1f}%")
print("=" * 60)