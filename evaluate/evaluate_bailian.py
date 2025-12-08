import json
import time
import os
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI  # 使用 openai 库的客户端，但我们将指向阿里云百炼


class RestaurantDataEvaluator:
    """餐厅推荐数据集质量评估器 (适配阿里云百炼平台)"""

    def __init__(self, api_key: str, model: str = "qwen-max"):
        """
        初始化评估器

        Args:
            api_key: 阿里云百炼的API Key，格式为 sk-xxxxxxxx。
            model: 在百炼平台选择的模型名称，例如 'qwen-max', 'qwen-plus'。
        """
        self.model = model

        # 初始化OpenAI客户端，将base_url指向阿里云百炼的兼容端点
        self.client = OpenAI(
            api_key=api_key,  # 使用你提供的密钥
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼OpenAI兼容接口地址[citation:9]
        )

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个数据样本

        Args:
            sample: 包含Instruction, Question, Answer的数据样本

        Returns:
            包含评分和理由的字典
        """
        # 1. 构建评估提示
        prompt = self._build_evaluation_prompt(sample)

        try:
            # 2. 调用LLM API进行评估[citation:2][citation:5]
            response_content = self._call_llm_api(prompt)

            # 3. 解析LLM的响应，提取评分
            evaluation = self._parse_evaluation_response(response_content)

            return {
                **sample,
                "training_data_quality_score": evaluation.get("training_data_quality_score"),
                "training_data_quality_reason": evaluation.get("training_data_quality_reason"),
                "original_model_performance_score": evaluation.get("original_model_performance_score"),
                "original_model_performance_reason": evaluation.get("original_model_performance_reason"),
            }

        except Exception as e:
            print(f"评估样本时出错: {e}")
            # 返回默认值
            return {
                **sample,
                "training_data_quality_score": None,
                "training_data_quality_reason": f"评估失败: {str(e)}",
                "original_model_performance_score": None,
                "original_model_performance_reason": f"评估失败: {str(e)}",
            }

    def evaluate_batch(self, samples: List[Dict[str, Any]],
                       batch_size: int = 5,
                       delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量评估数据样本，自动控制请求频率[citation:5]

        Args:
            samples: 数据样本列表
            batch_size: 批量处理大小（提示用，实际为逐条处理）
            delay: 每次API调用之间的延迟（秒），避免速率限制

        Returns:
            包含评分结果的样本列表
        """
        results = []
        total_samples = len(samples)

        print(f"开始评估 {total_samples} 个样本...")

        for i, sample in enumerate(samples):
            print(f"正在评估样本 {i + 1}/{total_samples}...")

            result = self.evaluate_sample(sample)
            results.append(result)

            # 添加延迟以避免API速率限制[citation:1]
            if i < total_samples - 1:
                time.sleep(delay)

        print("批量评估完成!")
        return results

    def _build_evaluation_prompt(self, sample: Dict[str, Any]) -> str:
        """构建评估提示"""

        prompt = f"""请你根据以下标准，对以下餐厅推荐数据样本进行训练数据质量和原始模型表现打分（均为10分制）：

**样本信息：**
Instruction: {sample.get('Instruction', '')}
Question: {sample.get('Question', '')}
Answer: {sample.get('Answer', '')}

**训练数据质量评分标准（10分制）：**
1. 准确性（2分）：餐厅信息是否真实、地址、评分、人均价格是否准确。
2. 一致性（2分）：相同或相似问题下的回答是否逻辑一致、信息不冲突。
3. 多样性（2分）：是否覆盖多种偏好、预算、位置、菜系组合。
4. 格式规范性（2分）：Instruction、Question、Answer 结构是否清晰、完整。
5. 实用性（2分）：推荐是否合理、是否真正符合用户偏好与预算。

**原始模型表现评分标准（10分制）：**
1. 理解能力（2分）：是否准确理解用户位置、预算、偏好。
2. 推荐合理性（2分）：推荐餐厅是否符合用户所有条件。
3. 信息完整性（2分）：是否提供名称、地址、评分、人均、距离等关键信息。
4. 逻辑清晰性（2分）：回答结构是否清晰、推荐理由是否充分。
5. 多样性（2分）：是否避免重复推荐，是否覆盖不同选择。

**打分要求：**
1. 训练数据质量评分（1-10分）：
   - 你的评分：[请在此处给出整数评分]
   - 简要理由：[请在此处简要说明理由，不超过50字]

2. 原始模型表现评分（1-10分）：
   - 你的评分：[请在此处给出整数评分]
   - 简要理由：[请在此处简要说明理由，不超过50字]

请严格按照以下JSON格式返回结果，不要返回任何其他文字：
{{
  "training_data_quality_score": 分数,
  "training_data_quality_reason": "理由",
  "original_model_performance_score": 分数,
  "original_model_performance_reason": "理由"
}}"""

        return prompt

    def _call_llm_api(self, prompt: str) -> str:
        """调用阿里云百炼的LLM API获取响应[citation:2]"""
        try:
            # 使用OpenAI兼容接口进行调用[citation:9]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "你是一个专业的AI训练数据质量评估专家。请严格按照评分标准和JSON格式要求进行评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度以获得更确定性的输出
                max_tokens=500
            )
            # 提取返回的文本内容
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"调用阿里云百炼API失败: {str(e)}。请检查API密钥是否正确、是否有额度、模型名称是否正确。")

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """解析LLM的响应，提取评分和理由"""
        try:
            # 尝试从响应中提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)

                # 验证必需的字段
                required_fields = [
                    "training_data_quality_score",
                    "training_data_quality_reason",
                    "original_model_performance_score",
                    "original_model_performance_reason"
                ]

                for field in required_fields:
                    if field not in evaluation:
                        raise ValueError(f"响应中缺少必需的字段: {field}")

                return evaluation
            else:
                # 如果没有找到JSON，尝试手动解析
                return self._parse_text_response(response)

        except json.JSONDecodeError:
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """解析文本格式的响应（备选方案）"""
        import re
        result = {
            "training_data_quality_score": None,
            "training_data_quality_reason": "无法解析响应格式",
            "original_model_performance_score": None,
            "original_model_performance_reason": "无法解析响应格式"
        }

        # 尝试查找训练数据质量评分
        tdq_patterns = [
            r'训练数据质量评分[：:]\s*(\d+)',
            r'training_data_quality_score[：:]\s*(\d+)',
            r'数据质量评分[：:]\s*(\d+)'
        ]

        for pattern in tdq_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["training_data_quality_score"] = int(match.group(1))
                break

        # 尝试查找原始模型表现评分
        omp_patterns = [
            r'原始模型表现评分[：:]\s*(\d+)',
            r'original_model_performance_score[：:]\s*(\d+)',
            r'模型表现评分[：:]\s*(\d+)'
        ]

        for pattern in omp_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["original_model_performance_score"] = int(match.group(1))
                break

        return result

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估结果，生成统计信息"""
        tdq_scores = [r.get("training_data_quality_score") for r in results if
                      r.get("training_data_quality_score") is not None]
        omp_scores = [r.get("original_model_performance_score") for r in results if
                      r.get("original_model_performance_score") is not None]

        # 计算统计数据
        analysis = {
            "total_samples": len(results),
            "training_data_quality": {
                "average_score": sum(tdq_scores) / len(tdq_scores) if tdq_scores else 0,
                "min_score": min(tdq_scores) if tdq_scores else 0,
                "max_score": max(tdq_scores) if tdq_scores else 0,
                "score_distribution": self._calculate_score_distribution(tdq_scores)
            },
            "original_model_performance": {
                "average_score": sum(omp_scores) / len(omp_scores) if omp_scores else 0,
                "min_score": min(omp_scores) if omp_scores else 0,
                "max_score": max(omp_scores) if omp_scores else 0,
                "score_distribution": self._calculate_score_distribution(omp_scores)
            },
            "correlation": self._calculate_correlation(tdq_scores, omp_scores)
        }

        return analysis

    def _calculate_score_distribution(self, scores: List[int]) -> Dict[int, int]:
        """计算评分分布"""
        distribution = {i: 0 for i in range(1, 11)}
        for score in scores:
            if 1 <= score <= 10:
                distribution[score] += 1
        return distribution

    def _calculate_correlation(self, scores1: List[int], scores2: List[int]) -> float:
        """计算两个评分序列的相关性"""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0

        try:
            import numpy as np
            return np.corrcoef(scores1, scores2)[0, 1]
        except:
            n = len(scores1)
            mean1 = sum(scores1) / n
            mean2 = sum(scores2) / n

            numerator = sum((s1 - mean1) * (s2 - mean2) for s1, s2 in zip(scores1, scores2))
            denominator = (sum((s1 - mean1) ** 2 for s1 in scores1) *
                           sum((s2 - mean2) ** 2 for s2 in scores2)) ** 0.5

            return numerator / denominator if denominator != 0 else 0.0


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载数据[citation:2]"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_results(results: List[Dict[str, Any]], file_path: str):
    """保存评估结果到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_analysis(analysis: Dict[str, Any], file_path: str):
    """保存分析结果到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)


def generate_report(analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """生成评估报告"""

    def format_distribution(distribution: Dict[int, int]) -> str:
        total = sum(distribution.values())
        result_lines = []
        for score in range(1, 11):
            count = distribution.get(score, 0)
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 5)
            result_lines.append(f"{score:2d}分: {count:3d}个 ({percentage:5.1f}%) {bar}")
        return "\n".join(result_lines)

    report = f"""餐厅推荐数据集质量评估报告 (阿里云百炼平台)
============================================
评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

基本信息
--------
评估样本总数: {analysis['total_samples']}

训练数据质量评估结果
------------------
平均分: {analysis['training_data_quality']['average_score']:.2f}
最低分: {analysis['training_data_quality']['min_score']}
最高分: {analysis['training_data_quality']['max_score']}

评分分布:
{format_distribution(analysis['training_data_quality']['score_distribution'])}

原始模型表现评估结果
------------------
平均分: {analysis['original_model_performance']['average_score']:.2f}
最低分: {analysis['original_model_performance']['min_score']}
最高分: {analysis['original_model_performance']['max_score']}

评分分布:
{format_distribution(analysis['original_model_performance']['score_distribution'])}

相关性分析
----------
训练数据质量与模型表现的相关性: {analysis['correlation']:.3f}
"""
    return report


def main():
    """主函数：执行完整的评估流程"""

    # ========== 配置参数 ==========
    # 1. 你的API密钥 (已直接填入)
    BAILIAN_API_KEY = "sk-79030d2fb6124a008441f6d9e3d88a9f"

    # 2. 选择模型：qwen-max, qwen-plus, qwen-turbo 等
    MODEL_NAME = "qwen-max"

    # 3. 文件路径
    DATA_FILE = "prediction_result.json"  # 输入数据文件
    RESULTS_FILE = "evaluation_results.json"  # 详细评估结果
    ANALYSIS_FILE = "evaluation_analysis.json"  # 统计分析结果
    REPORT_FILE = "evaluation_report.txt"  # 可读报告

    # 4. 评估控制
    SAMPLE_LIMIT = None  # 为测试先评估前10个样本，设为None则评估全部
    REQUEST_DELAY = 2.0  # 请求间隔(秒)，避免限流

    # ========== 执行评估 ==========
    print("=" * 60)
    print("餐厅推荐数据集质量评估系统 (阿里云百炼版)")
    print("=" * 60)

    # 1. 创建评估器
    print(f"初始化评估器，使用模型: {MODEL_NAME}")
    evaluator = RestaurantDataEvaluator(api_key=BAILIAN_API_KEY, model=MODEL_NAME)

    # 2. 加载数据
    print(f"从 '{DATA_FILE}' 加载数据...")
    try:
        all_data = load_data(DATA_FILE)
        print(f"成功加载 {len(all_data)} 个样本")

        # 限制评估样本数量（用于测试）
        if SAMPLE_LIMIT and SAMPLE_LIMIT < len(all_data):
            data_to_evaluate = all_data[:SAMPLE_LIMIT]
            print(f"为快速测试，将只评估前 {SAMPLE_LIMIT} 个样本")
        else:
            data_to_evaluate = all_data
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{DATA_FILE}'，请确保它和脚本在同一目录。")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件 '{DATA_FILE}' 格式不正确，不是有效的JSON。")
        return

    # 3. 批量评估
    print(f"开始评估 {len(data_to_evaluate)} 个样本，每次请求间隔 {REQUEST_DELAY} 秒...")
    results = evaluator.evaluate_batch(data_to_evaluate, delay=REQUEST_DELAY)

    # 4. 保存详细结果
    print(f"保存详细评估结果到 '{RESULTS_FILE}'...")
    save_results(results, RESULTS_FILE)

    # 5. 分析结果
    print("分析评估结果...")
    analysis = evaluator.analyze_results(results)

    # 6. 保存分析结果
    print(f"保存统计分析到 '{ANALYSIS_FILE}'...")
    save_analysis(analysis, ANALYSIS_FILE)

    # 7. 生成并保存报告
    print(f"生成评估报告到 '{REPORT_FILE}'...")
    report = generate_report(analysis, results)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    # 8. 打印摘要到屏幕
    print("\n" + "=" * 60)
    print("评估完成！摘要如下：")
    print(f"训练数据质量平均分: {analysis['training_data_quality']['average_score']:.2f}")
    print(f"原始模型表现平均分: {analysis['original_model_performance']['average_score']:.2f}")
    if analysis['correlation'] > 0.5:
        corr_msg = "强正相关"
    elif analysis['correlation'] > 0.3:
        corr_msg = "中等正相关"
    elif analysis['correlation'] > 0:
        corr_msg = "弱正相关"
    else:
        corr_msg = "无显著相关"
    print(f"两者相关性: {analysis['correlation']:.3f} ({corr_msg})")
    print("=" * 60)
    print(f"\n详细结果已保存至：")
    print(f"  - 详细评分: {RESULTS_FILE}")
    print(f"  - 统计分析: {ANALYSIS_FILE}")
    print(f"  - 评估报告: {REPORT_FILE}")

    # 9. 显示前3个样本的评估结果
    print("\n前3个样本的评估结果示例：")
    for i, result in enumerate(results[:3]):
        print(f"\n样本 {i + 1}:")
        print(f"  问题: {result['Question'][:60]}...")
        print(
            f"  训练数据质量: {result['training_data_quality_score']}分 - {result['training_data_quality_reason'][:30]}...")
        print(
            f"  原始模型表现: {result['original_model_performance_score']}分 - {result['original_model_performance_reason'][:30]}...")


if __name__ == "__main__":
    main()