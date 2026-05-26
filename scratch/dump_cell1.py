import json

file_path = r"d:\Unknown\Papper\Code\Ref_CODE\純SSD配對交易-[版本1].ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Cell 1 code:")
print("".join(nb['cells'][1].get('source', [])))
