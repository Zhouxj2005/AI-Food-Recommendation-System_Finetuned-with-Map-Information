from smolagents import Tool
import requests
import os
import requests

class RestaurantSearchTool(Tool):
    name = "restaurant_search"
    description = (
        "根据经纬度搜索附近餐厅。输入: location: 'lng,lat'; radius: int（默认2000米）"
    )
    inputs = {
        "location": {"type": "string", "description": "The location to search for restaurants around."},
        "radius": {"type": "integer", "description": "The radius to search for restaurants around.", "nullable": True}
    }
    output_type  = "string"

    def forward(self, location: str, radius: int = 2000):
        GAODE_KEY = "c81295d3a4d1b4f85ddf4bf082674de1"
        url = "https://restapi.amap.com/v3/place/around"
        params = {
            "key": GAODE_KEY,
            "location": location,
            "radius": radius,
            "types": "050000",  # 餐饮类
            "offset": 40,
            "page": 1
        }

        try:
            res = requests.get(url, params=params, timeout=5).json()
            if res["status"] == "1":
                return {"success": True, "restaurants": res["pois"]}
            else:
                return {"success": False, "error": res.get("info", "unknown error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
