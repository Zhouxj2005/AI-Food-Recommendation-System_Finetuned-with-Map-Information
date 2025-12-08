import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_evaluation_data(file_path):
    """加载评估数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_trained_model_scores(data):
    """计算训练后模型评分（同权重合并）"""
    trained_scores = []

    for item in data:
        # 获取原始评分
        train_score = item.get('training_data_quality_score', 0)
        model_score = item.get('original_model_performance_score', 0)

        # 同权重合并（简单平均）
        combined_score = (train_score + model_score) / 2

        # 四舍五入到整数或保留一位小数
        rounded_score = round(combined_score, 1)

        # 添加到列表中
        trained_scores.append(rounded_score)

        # 也可以在数据中添加新字段
        item['trained_model_score'] = rounded_score

    return trained_scores, data


def analyze_scores(scores):
    """分析评分数据"""
    # 转换为浮点数以确保统计正确
    scores_float = [float(score) for score in scores]

    # 基本统计
    avg_score = np.mean(scores_float)
    min_score = np.min(scores_float)
    max_score = np.max(scores_float)

    # 按0.5为区间统计分布
    bins = np.arange(0, 10.5, 0.5)
    hist, bin_edges = np.histogram(scores_float, bins=bins)

    # 按整数统计（便于展示）
    int_bins = list(range(1, 11))
    int_scores = [round(score) for score in scores_float]
    int_counter = Counter(int_scores)

    # 计算百分比
    total_count = len(scores_float)
    int_percentages = {score: count / total_count * 100 for score, count in int_counter.items()}

    return {
        'avg_score': avg_score,
        'min_score': min_score,
        'max_score': max_score,
        'total_count': total_count,
        'scores_float': scores_float,
        'int_scores': int_scores,
        'int_counter': int_counter,
        'int_percentages': int_percentages,
        'hist': hist,
        'bin_edges': bin_edges
    }


def create_visualization(analysis_results, original_data=None, save_path='trained_model_scores_visualization.png'):
    """创建训练后模型评分可视化图表"""

    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('训练后模型评分分析\n(训练数据质量与原始模型表现同权重合并)',
                 fontsize=16, fontweight='bold', y=1.02)

    # 数据准备
    scores_float = analysis_results['scores_float']
    int_counter = analysis_results['int_counter']
    int_percentages = analysis_results['int_percentages']
    hist = analysis_results['hist']
    bin_edges = analysis_results['bin_edges']

    # 1. 评分分布直方图（按0.5分间隔）
    ax1.hist(scores_float, bins=bin_edges, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_title('训练后模型评分分布（0.5分间隔）', fontsize=14, fontweight='bold')
    ax1.set_xlabel('评分', fontsize=12)
    ax1.set_ylabel('样本数量', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # 添加平均值线
    ax1.axvline(analysis_results['avg_score'], color='red', linestyle='--',
                linewidth=2, label=f'平均分: {analysis_results["avg_score"]:.2f}')
    ax1.legend()

    # 2. 评分分布条形图（按整数）
    int_scores_sorted = sorted(int_counter.keys())
    int_counts = [int_counter[score] for score in int_scores_sorted]
    int_percents = [int_percentages.get(score, 0) for score in int_scores_sorted]

    bars = ax2.bar(int_scores_sorted, int_counts, edgecolor='black', alpha=0.7, color='lightcoral')
    ax2.set_title('训练后模型评分分布（整数评分）', fontsize=14, fontweight='bold')
    ax2.set_xlabel('评分', fontsize=12)
    ax2.set_ylabel('样本数量', fontsize=12)
    ax2.set_xticks(int_scores_sorted)
    ax2.grid(axis='y', alpha=0.3)

    # 在柱子上添加标签
    for bar, count, percent in zip(bars, int_counts, int_percents):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{count}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=9)

    # 3. 评分密度曲线
    from scipy.stats import gaussian_kde

    # 创建密度估计
    if len(scores_float) > 1:
        density = gaussian_kde(scores_float)
        xs = np.linspace(min(scores_float) - 0.5, max(scores_float) + 0.5, 200)
        ys = density(xs)

        ax3.plot(xs, ys, color='darkgreen', linewidth=2)
        ax3.fill_between(xs, ys, alpha=0.3, color='lightgreen')

        # 标记平均值
        ax3.axvline(analysis_results['avg_score'], color='red', linestyle='--',
                    linewidth=2, label=f'平均分: {analysis_results["avg_score"]:.2f}')

        # 标记中位数
        median_score = np.median(scores_float)
        ax3.axvline(median_score, color='blue', linestyle=':',
                    linewidth=2, label=f'中位数: {median_score:.2f}')

        ax3.set_title('训练后模型评分密度分布', fontsize=14, fontweight='bold')
        ax3.set_xlabel('评分', fontsize=12)
        ax3.set_ylabel('密度', fontsize=12)
        ax3.grid(alpha=0.3)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '数据不足\n无法计算密度分布',
                 ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_title('训练后模型评分密度分布', fontsize=14, fontweight='bold')

    # 4. 评分统计分析
    ax4.axis('off')

    # 计算更多统计信息
    std_score = np.std(scores_float)
    median_score = np.median(scores_float)

    # 计算不同分数段占比
    excellent = len([s for s in scores_float if s >= 8]) / len(scores_float) * 100
    good = len([s for s in scores_float if 6.5 <= s < 8]) / len(scores_float) * 100
    fair = len([s for s in scores_float if 5 <= s < 6.5]) / len(scores_float) * 100
    poor = len([s for s in scores_float if s < 5]) / len(scores_float) * 100

    # 与原始数据对比（如果提供了原始数据）
    if original_data:
        # 提取原始评分
        train_scores = [item.get('training_data_quality_score', 0) for item in original_data]
        model_scores = [item.get('original_model_performance_score', 0) for item in original_data]

        train_avg = np.mean(train_scores)
        model_avg = np.mean(model_scores)

        comparison_text = f"""
与原始评分对比:
• 训练数据平均分: {train_avg:.2f}
• 原始模型平均分: {model_avg:.2f}
• 训练后模型平均分: {analysis_results['avg_score']:.2f}
• 提升幅度: {analysis_results['avg_score'] - model_avg:.2f}
        """
    else:
        comparison_text = ""

    stats_text = f"""
📊 训练后模型评分统计

基本统计:
• 样本总数: {analysis_results['total_count']}
• 平均分: {analysis_results['avg_score']:.2f}
• 中位数: {median_score:.2f}
• 标准差: {std_score:.2f}
• 最低分: {analysis_results['min_score']:.2f}
• 最高分: {analysis_results['max_score']:.2f}

📈 分数段分布:
• 优秀 (≥8.0): {excellent:.1f}%
• 良好 (6.5-7.9): {good:.1f}%
• 一般 (5.0-6.4): {fair:.1f}%
• 较差 (<5.0): {poor:.1f}%

🔍 主要特征:
• 主要分布区间: {np.percentile(scores_float, 25):.1f} - {np.percentile(scores_float, 75):.1f}
• 变异系数: {(std_score / analysis_results['avg_score'] * 100):.1f}%

{comparison_text}

✅ 结论:
训练后模型评分综合了数据质量和原始表现，
平均分{analysis_results['avg_score']:.2f}，
{excellent:.1f}%的样本达到优秀水平。
"""

    ax4.text(0.05, 0.5, stats_text, fontsize=11,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.7, edgecolor='gold', linewidth=2))

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def save_enhanced_data(data, output_path='enhanced_evaluation_result.json'):
    """保存增强后的数据（包含训练后模型评分）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"增强后的数据已保存到: {output_path}")


def main():
    """主函数"""
    # 文件路径
    json_file = 'evaluation_results.json'

    try:
        # 1. 加载数据
        print("正在加载数据...")
        data = load_evaluation_data(json_file)
        print(f"成功加载 {len(data)} 条记录")

        # 2. 计算训练后模型评分
        print("正在计算训练后模型评分...")
        trained_scores, enhanced_data = calculate_trained_model_scores(data)

        # 3. 分析评分
        print("正在分析评分数据...")
        analysis_results = analyze_scores(trained_scores)

        # 4. 创建可视化
        print("正在生成可视化图表...")
        fig = create_visualization(analysis_results, data)

        # 5. 保存增强后的数据
        save_enhanced_data(enhanced_data)

        # 6. 打印详细统计
        print("\n" + "=" * 70)
        print("训练后模型评分详细统计")
        print("=" * 70)
        print(f"平均分: {analysis_results['avg_score']:.2f}")
        print(f"中位数: {np.median(analysis_results['scores_float']):.2f}")
        print(f"标准差: {np.std(analysis_results['scores_float']):.2f}")
        print(f"评分范围: {analysis_results['min_score']:.2f} - {analysis_results['max_score']:.2f}")
        print()

        print("整数评分分布:")
        for score in sorted(analysis_results['int_counter'].keys()):
            count = analysis_results['int_counter'][score]
            percent = analysis_results['int_percentages'].get(score, 0)
            print(f"  {score}分: {count}个 ({percent:.1f}%)")

        print("\n" + "=" * 70)

    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file}")
        print("请确保 evaluation_result.json 文件在当前目录下")
    except json.JSONDecodeError:
        print(f"错误: {json_file} 文件格式不正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")


# 如果直接运行此脚本，执行主函数
if __name__ == "__main__":
    main()