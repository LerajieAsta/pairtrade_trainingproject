import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/strategies_logic.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, cell in enumerate(data["cells"]):
    cell_type = cell["cell_type"]
    source = cell["source"]
    source_len = len(source)
    first_few_chars = "".join(source[:2])[:100].replace("\n", " ")
    print(f"Cell [{i:02d}] Type: {cell_type:<8} Lines: {source_len:<4} Preview: {first_few_chars}")
