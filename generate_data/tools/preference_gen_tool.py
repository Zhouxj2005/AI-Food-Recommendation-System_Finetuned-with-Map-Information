from smolagents import Tool

class PreferenceGenTool(Tool):
    name = "preference_gen"
    description = "随机生成用户偏好，包括口味、预算、风格"
    inputs = {}
    output_type  = "string"

    def forward(self):
        import random

        cuisine = random.choice([
            "川菜", "粤菜", "湘菜", "东北菜", "江浙菜", "日本料理", "烧烤", "火锅", "西餐"
        ])
        style = random.choice([
            "喜欢辣", "清淡口味", "重油重盐", "健康饮食", "偏爱甜食", "偏爱肉类", "无辣不欢"
        ])
        budget = random.choice(["人均50以内", "人均50-100", "人均100-200", "人均200以上"])

        return {
            "cuisine": cuisine,
            "style": style,
            "budget": budget
        }
