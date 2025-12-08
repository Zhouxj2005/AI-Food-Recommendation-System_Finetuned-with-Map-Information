import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
scores = list(range(1, 11))

# 训练数据质量分布
train_counts = [0, 1, 1, 38, 34, 137, 148, 530, 201, 0]
train_percentages = [0.0, 0.1, 0.1, 3.5, 3.1, 12.6, 13.6, 48.6, 18.4, 0.0]

# 原始模型表现分布
model_counts = [1, 1, 37, 34, 136, 146, 497, 194, 44, 0]
model_percentages = [0.1, 0.1, 3.4, 3.1, 12.5, 13.4, 45.6, 17.8, 4.0, 0.0]

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('餐厅推荐数据集质量评估报告可视化分析\n(评估时间: 2025-12-08 12:31:44)',
             fontsize=16, fontweight='bold', y=1.02)

# 1. 训练数据质量评分分布 (柱状图)
colors_train = ['lightgray', 'lightcoral', 'lightcoral', 'lightsalmon', 'lightsalmon',
                'lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightgray']
bars1 = ax1.bar(scores, train_counts, color=colors_train, edgecolor='black', alpha=0.8)
ax1.set_title('训练数据质量评分分布', fontsize=14, fontweight='bold')
ax1.set_xlabel('评分', fontsize=12)
ax1.set_ylabel('样本数量', fontsize=12)
ax1.set_xticks(scores)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(train_counts) * 1.15)

# 在柱子上添加标签
for i, (bar, count, perc) in enumerate(zip(bars1, train_counts, train_percentages)):
    if count > 0:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}\n({perc}%)', ha='center', va='bottom', fontsize=9)

# 2. 原始模型表现评分分布 (柱状图)
colors_model = ['lightcoral', 'lightcoral', 'lightcoral', 'lightsalmon', 'lightsalmon',
                'lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightgray']
bars2 = ax2.bar(scores, model_counts, color=colors_model, edgecolor='black', alpha=0.8)
ax2.set_title('原始模型表现评分分布', fontsize=14, fontweight='bold')
ax2.set_xlabel('评分', fontsize=12)
ax2.set_ylabel('样本数量', fontsize=12)
ax2.set_xticks(scores)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(model_counts) * 1.15)

# 在柱子上添加标签
for i, (bar, count, perc) in enumerate(zip(bars2, model_counts, model_percentages)):
    if count > 0:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}\n({perc}%)', ha='center', va='bottom', fontsize=9)

# 3. 并列对比柱状图
ax3 = plt.subplot(2, 2, 3)
width = 0.35
x = np.arange(len(scores))

bars3_train = ax3.bar(x - width/2, train_counts, width, label='训练数据质量',
                      color='steelblue', alpha=0.7, edgecolor='black')
bars3_model = ax3.bar(x + width/2, model_counts, width, label='模型表现',
                      color='indianred', alpha=0.7, edgecolor='black')

ax3.set_title('训练数据与模型表现对比', fontsize=14, fontweight='bold')
ax3.set_xlabel('评分', fontsize=12)
ax3.set_ylabel('样本数量', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(scores)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 添加差异标注
for i, (train_count, model_count) in enumerate(zip(train_counts, model_counts)):
    diff = train_count - model_count
    if abs(diff) > 20:  # 只在差异较大时标注
        y_pos = max(train_count, model_count) + 15
        ax3.text(i, y_pos, f'{diff:+d}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='green' if diff > 0 else 'red')

# 4. 关键指标和趋势分析
ax4.axis('off')

# 计算统计数据
train_avg = 7.55
model_avg = 6.64
correlation = 0.956
train_high_percent = (530 + 201) / 1090 * 100  # 8-9分比例
model_high_percent = (497 + 194) / 1090 * 100  # 7-8分比例

metrics_text = f"""
关键指标汇总

📊 数据集信息:
• 样本总数: 1090
• 评估时间: 2025-12-08 12:31:44

🏆 训练数据质量:
• 平均分: {train_avg}
• 最低分: 2
• 最高分: 9
• 主要分布: 8分 ({train_percentages[7]}%)
• 8-9分占比: {train_high_percent:.1f}%

🤖 原始模型表现:
• 平均分: {model_avg}
• 最低分: 1
• 最高分: 9
• 主要分布: 7分 ({model_percentages[6]}%)
• 7-8分占比: {model_high_percent:.1f}%

📈 相关性分析:
• 训练数据与模型表现相关性: {correlation}

📋 主要发现:
1. 训练数据质量保持较高水平
   - 平均分: {train_avg} (与前次7.58分基本持平)
   - 8-9分占比: {train_high_percent:.1f}% (质量集中)

2. 模型表现稳定
   - 平均分: {model_avg} (与前次6.66分基本持平)
   - 7-8分占比: {model_high_percent:.1f}% (表现集中)

3. 高度相关关系
   - 相关性系数: {correlation} (接近完美正相关)
   - 表明数据质量直接影响模型表现
"""

ax4.text(0.05, 0.5, metrics_text, fontsize=11,
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                  alpha=0.7, edgecolor='gold', linewidth=2))

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('evaluation_visualization_v2.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细对比分析
print("=" * 70)
print("餐厅推荐数据集质量评估报告 - 详细分析")
print("=" * 70)
print(f"评估时间: 2025-12-08 12:31:44")
print(f"样本总数: 1090")
print()
print("训练数据质量分析:")
print(f"  • 平均分: {train_avg}")
print(f"  • 分数分布: 主要集中在8分({train_percentages[7]}%)和9分({train_percentages[8]}%)")
print(f"  • 高质量数据(8-9分): {train_high_percent:.1f}%")
print()
print("模型表现分析:")
print(f"  • 平均分: {model_avg}")
print(f"  • 分数分布: 主要集中在7分({model_percentages[6]}%)和8分({model_percentages[7]}%)")
print(f"  • 良好表现(7-8分): {model_high_percent:.1f}%")
print()
print("对比分析:")
print(f"  • 训练数据平均分 > 模型表现平均分: {train_avg - model_avg:.2f}")
print(f"  • 相关性系数: {correlation} (高度正相关)")
print(f"  • 结论: 数据质量对模型表现有显著影响，提升数据质量可进一步提升模型表现")
print("=" * 70)