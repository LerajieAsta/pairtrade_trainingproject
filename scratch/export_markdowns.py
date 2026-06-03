import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/strategies_logic.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("scratch/strategies_logic_markdown.md", "w", encoding="utf-8") as out:
    for i, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "markdown":
            out.write(f"\n\n<!-- ==================== CELL {i} ==================== -->\n\n")
            out.write("".join(cell["source"]))
        elif cell["cell_type"] == "code":
            out.write(f"\n\n<!-- ==================== CODE CELL {i} ==================== -->\n\n")
            # 寫出代碼的前 5 行與後 5 行，並標記中間被省略
            lines = cell["source"]
            if len(lines) <= 20:
                out.write("```python\n" + "".join(lines) + "\n```")
            else:
                out.write("```python\n" + "".join(lines[:10]) + "\n... [OMITTED] ...\n" + "".join(lines[-10:]) + "\n```")
