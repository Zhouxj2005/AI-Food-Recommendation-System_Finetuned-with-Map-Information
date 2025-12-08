import json
import time
import os
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI


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
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个数据样本

        Args:
            sample: 包含Instruction, Question, Answer的数据样本

        Returns:
            包含综合评分的字典
        """
        # 1. 构建评估提示
        prompt = self._build_evaluation_prompt(sample)

        try:
            # 2. 调用LLM API进行评估
            response_content = self._call_llm_api(prompt)

            # 3. 解析LLM的响应，提取综合评分
            evaluation = self._parse_evaluation_response(response_content)

            return {
                **sample,
                "composite_score": evaluation.get("composite_score"),
                "composite_score_reason": evaluation.get("composite_score_reason"),
            }

        except Exception as e:
            print(f"评估样本时出错: {e}")
            # 返回默认值
            return {
                **sample,
                "composite_score": None,
                "composite_score_reason": f"评估失败: {str(e)}",
            }

    def evaluate_batch(self, samples: List[Dict[str, Any]],
                       batch_size: int = 5,
                       delay: float = 1.0) -> List[Dict[str, Any]]:
        """
        批量评估数据样本，自动控制请求频率

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

            # 添加延迟以避免API速率限制
            if i < total_samples - 1:
                time.sleep(delay)

        print("批量评估完成!")
        return results

    def _build_evaluation_prompt(self, sample: Dict[str, Any]) -> str:
        """构建综合评估提示"""

        prompt = f"""请你根据以下综合标准，对以下餐厅推荐数据样本进行综合打分（10分制）：

**样本信息：**
Instruction: {sample.get('Instruction', '')}
Question: {sample.get('Question', '')}
Answer: {sample.get('Answer', '')}

**综合评分标准（10分制）：**
1. 数据质量与准确性（2分）：Instruction、Question、Answer是否清晰准确，信息是否真实可靠。
2. 推荐合理性（2分）：推荐餐厅是否完全符合用户的位置、预算、偏好等所有条件。
3. 信息完整性（2分）：是否提供完整的餐厅信息（名称、地址、评分、人均价格、距离等）。
4. 多样性与实用性（2分）：是否提供多种选择，推荐是否实际可用，是否覆盖不同需求。
5. 逻辑清晰性（2分）：回答结构是否清晰，推荐理由是否充分，逻辑是否一致。

**打分要求：**
综合评分（1-10分）：
   - 你的评分：[请在此处给出整数评分]
   - 简要理由：[请在此处简要说明理由，不超过50字]

请严格按照以下JSON格式返回结果，不要返回任何其他文字：
{{
  "composite_score": 分数,
  "composite_score_reason": "理由"
}}"""

        return prompt

    def _call_llm_api(self, prompt: str) -> str:
        """调用阿里云百炼的LLM API获取响应"""
        try:
            # 使用OpenAI兼容接口进行调用
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
        """解析LLM的响应，提取综合评分和理由"""
        try:
            # 尝试从响应中提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                evaluation = json.loads(json_str)

                # 验证必需的字段
                required_fields = ["composite_score", "composite_score_reason"]

                for field in required_fields:
                    if field not in evaluation:
                        raise ValueError(f"响应中缺少必需的字段: {field}")

                # 确保评分在有效范围内
                score = evaluation["composite_score"]
                if not isinstance(score, (int, float)) or score < 1 or score > 10:
                    raise ValueError(f"评分必须在1-10之间，当前为: {score}")

                return evaluation
            else:
                # 如果没有找到JSON，尝试手动解析
                return self._parse_text_response(response)

        except json.JSONDecodeError:
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """解析文本格式的响应（备选方案）"""
        result = {
            "composite_score": 5,  # 默认中等分数
            "composite_score_reason": "无法解析响应格式，使用默认评分"
        }

        # 尝试查找综合评分
        patterns = [
            r'综合评分[：:]\s*(\d+)',
            r'composite_score[：:]\s*(\d+)',
            r'评分[：:]\s*(\d+)',
            r'分数[：:]\s*(\d+)',
            r'score[：:]\s*(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 10:
                        result["composite_score"] = score
                        # 尝试提取理由
                        reason_match = re.search(r'理由[：:]\s*(.*?)(?=\n|$)', response, re.IGNORECASE)
                        if reason_match:
                            result["composite_score_reason"] = reason_match.group(1).strip()
                    break
                except ValueError:
                    continue

        return result

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析评估结果，生成统计信息"""
        composite_scores = [r.get("composite_score") for r in results
                          if r.get("composite_score") is not None]

        # 计算统计数据
        analysis = {
            "total_samples": len(results),
            "evaluated_samples": len(composite_scores),
            "composite_scores": {
                "average_score": sum(composite_scores) / len(composite_scores) if composite_scores else 0,
                "min_score": min(composite_scores) if composite_scores else 0,
                "max_score": max(composite_scores) if composite_scores else 0,
                "score_distribution": self._calculate_score_distribution(composite_scores)
            }
        }

        return analysis

    def _calculate_score_distribution(self, scores: List[int]) -> Dict[int, int]:
        """计算评分分布"""
        distribution = {i: 0 for i in range(1, 11)}
        for score in scores:
            if 1 <= score <= 10:
                distribution[score] += 1
        return distribution


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载数据"""
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
    """生成简化的评估报告"""

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
有效评估样本: {analysis['evaluated_samples']}

综合评分评估结果
------------------
平均分: {analysis['composite_scores']['average_score']:.2f}
最低分: {analysis['composite_scores']['min_score']}
最高分: {analysis['composite_scores']['max_score']}

评分分布:
{format_distribution(analysis['composite_scores']['score_distribution'])}
"""
    return report


def main():
    """主函数：执行完整的评估流程"""

    # ========== 配置参数 ==========
    # 1. 你的API密钥
    BAILIAN_API_KEY = "sk-79030d2fb6124a008441f6d9e3d88a9f"
    # 1. 你的API密钥 (已直接填入)
    BAILIAN_API_KEY = "sk-46840bdaeb31444a80dce5444af61633"

    # 2. 选择模型：qwen-max, qwen-plus, qwen-turbo 等
    MODEL_NAME = "qwen-max"

    # 3. 文件路径
    DATA_FILE = "data.json"  # 输入数据文件
    RESULTS_FILE = "composite_evaluation_results.json"  # 详细评估结果
    ANALYSIS_FILE = "composite_evaluation_analysis.json"  # 统计分析结果
    REPORT_FILE = "composite_evaluation_report.txt"  # 可读报告
    DATA_FILE = "./raw_mode_evaluation/raw_model_dataset.json"  # 输入数据文件
    RESULTS_FILE = "./raw_mode_evaluation/evaluation_results.json"  # 详细评估结果
    ANALYSIS_FILE = "./raw_mode_evaluation/evaluation_analysis.json"  # 统计分析结果
    REPORT_FILE = "./raw_mode_evaluation/evaluation_report.txt"  # 可读报告

    # 4. 评估控制
    SAMPLE_LIMIT = None  # 为测试先评估前N个样本，设为None则评估全部
    REQUEST_DELAY = 2.0  # 请求间隔(秒)，避免限流

    # ========== 执行评估 ==========
    print("=" * 60)
    print("餐厅推荐数据集质量评估系统 - 综合评分版")
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
    print(f"综合评分平均分: {analysis['composite_scores']['average_score']:.2f}")
    print(f"评分范围: {analysis['composite_scores']['min_score']} - {analysis['composite_scores']['max_score']}")
    print(f"有效评估样本: {analysis['evaluated_samples']}/{analysis['total_samples']}")
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
        print(f"  综合评分: {result['composite_score']}分")
        print(f"  理由: {result['composite_score_reason'][:50]}...")


if __name__ == "__main__":
    main()