from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent
from tools.geo_location_tool import GeoLocationTool
from tools.restaurant_search_tool import RestaurantSearchTool
from tools.preference_gen_tool import PreferenceGenTool

geo_location = GeoLocationTool()
restaurant_search = RestaurantSearchTool()
preference_gen = PreferenceGenTool()

class DataGeneratorAgent(ToolCallingAgent):

    def __init__(self):
        super().__init__(
            model = LiteLLMModel(
                model_id="dashscope/qwen3-max",
                api_key="sk-79030d2fb6124a008441f6d9e3d88a9f",
                api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            tools=[
                geo_location,
                restaurant_search,
                preference_gen
            ]
        )

    def generate_one(self, place_name: str,nums: int = 5):
        prompt = f"""
        你是一个美食推荐数据生成器。
        用户位置：{place_name}

        请：
        1. 调用 geo_location 获取经纬度
        2. 调用 restaurant_search 获取附近餐厅
        3. 调用 preference_gen 生成用户偏好
        4. 根据生成的偏好和搜索到的餐厅信息，综合分析得到推荐的餐厅，并将信息整理成下述JSON格式：
        {{
        "Instruction": "根据用户位置和偏好，推荐符合要求的餐厅",
        "Question": "..."(口语化描述用户的需求，例如："我在北京良乡北京理工大学，价格需求区间为50-100，偏好川菜、健康饮食"),
        "Answer":"..."(口语化描述推荐的餐厅信息，包括名称、地址、理由(评分、距离等)，除非没有足够满足要求的餐厅，否则至少推荐3个餐厅)
        }}
        5. 重复3、4，生成{nums}个样例
        只输出 JSON，不要任何解释。
        下面是一个根据偏好分析、推荐餐厅的示例（并不是最后的输出，需要将信息整理成上述JSON格式）：
            "基本信息":
                "个人预算": 200,
                "饮食偏好": ["辣", "中式快餐", "面食"],
                "住址": "北京市海淀区中关村大街123号",
                "理想通勤时间": 40
            "推荐餐厅": [
                    "餐厅名称": "川香源",
                    "菜系": "川菜",
                    "推荐理由": "符合您的辣味偏好，且距离较近",
                    "地址": "海淀区文慧园西路36号",
                    "预估通勤时间": 23,
                    "人均消费": 76,
                    "推荐菜品": ["香辣烤鱼", "辣子鸡", "香辣烤牛蛙"]
                ,
                    "餐厅名称": "方砖场69号炸酱面(中关村大融城店)",
                    "菜系": "京菜",
                    "推荐理由": "您喜爱的面食，地道老北京风味",
                    "地址": "北京市海淀区中关村大街15号",
                    "预估通勤时间": 25,
                    "人均消费": 42,
                    "推荐菜品": ["我们的炸酱面", "老北京爆肚粉", "老北京炸酱面"]
            ]
                """

        return self.run(prompt)
