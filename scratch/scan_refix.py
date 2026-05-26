import json
import sys

file_path = r"d:\Unknown\Papper\Code\Ref_CODE\純SSD配對交易_Refix.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

out_lines = []
out_lines.append(f"Total cells in Refix: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type', '')
    outputs = cell.get('outputs', [])
    source = "".join(cell.get('source', []))
    
    if outputs:
        for o_idx, out in enumerate(outputs):
            out_type = out.get('output_type', '')
            if out_type == 'stream':
                text = "".join(out.get('text', []))
                out_lines.append(f"Cell {idx} has stream output:")
                out_lines.append(text[:2000])
            elif out_type in ['execute_result', 'display_data']:
                data = out.get('data', {})
                for mime, val in data.items():
                    if mime == 'text/plain':
                        val_str = "".join(val)
                        out_lines.append(f"Cell {idx} has plain text output:")
                        out_lines.append(val_str[:2000])
                    elif mime == 'text/html':
                        val_str = "".join(val)
                        out_lines.append(f"Cell {idx} has HTML output (length {len(val_str)})")

with open(r"d:\Unknown\Papper\Code\scratch\scan_refix_output.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))
print("Successfully wrote scan results to file!")
