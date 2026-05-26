import json

file_path = r"d:\Unknown\Papper\Code\Ref_CODE\純SSD配對交易_Refix.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = "".join(nb['cells'][1].get('source', []))

with open(r"d:\Unknown\Papper\Code\scratch\dump_refix_cell1_output.txt", 'w', encoding='utf-8') as f:
    f.write(source)
print("Successfully wrote Refix Cell 1 code to file!")
