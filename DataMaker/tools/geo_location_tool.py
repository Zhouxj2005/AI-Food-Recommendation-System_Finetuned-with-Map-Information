from smolagents import Tool
import requests
import os
import json



class GeoLocationTool(Tool):
    name = "geo_location"
    description = "根据地名（如：上海市静安区）查询经纬度坐标。输入参数：place_name: str"
    inputs = {"place_name": {"type": "string", "description": "The name of the place to get the location for."}}
    output_type  = "string"


    def forward(self, place_name: str):
        GAODE_KEY = "c81295d3a4d1b4f85ddf4bf082674de1"
        url = "https://restapi.amap.com/v3/geocode/geo"
        params = {"address": place_name, "key": GAODE_KEY}

        try:
            res = requests.get(url, params=params, timeout=5).json()
            if res["status"] == "1" and res["geocodes"]:
                return {"success": True, "location": res["geocodes"][0]["location"]}
            else:
                return {"success": False, "error": res.get("info", "unknown error")}
        except Exception as e:
            return {"success": False, "error": str(e)}

# if __name__ == "__main__":
#     ge = GeoLocationTool()
#     res = ge.forward("上海市静安区")
#     print(res)
