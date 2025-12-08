import json

merged = []

with open("data_raw.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        try:
            items = json.loads(line)
        except Exception as e:
            print(f"跳过错误行: {line[:50]}... 错误: {e}")
            continue

        if isinstance(items, list):
            merged.extend(items)
        else:
            merged.append(items)

with open("merged2.json", "w", encoding="utf-8") as f:
    f.write("[\n")
    for i, obj in enumerate(merged):
        f.write("    ")
        json_line = json.dumps(obj, ensure_ascii=False)
        if i < len(merged) - 1:
            json_line += ","
        f.write(json_line + "\n")
    f.write("]\n")

print(f"写入完成，共 {len(merged)} 条，输出文件：merged.json")