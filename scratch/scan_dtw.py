import json

file_path = r"d:\Unknown\Papper\Code\Ref_CODE\純DTW配對交易-[版本2].ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

out_lines = []
out_lines.append(f"Total cells in DTW Notebook: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type', '')
    outputs = cell.get('outputs', [])
    source = "".join(cell.get('source', []))
    
    out_lines.append(f"\n======================================")
    out_lines.append(f"Cell {idx} ({cell_type})")
    out_lines.append(f"======================================")
    out_lines.append("CODE PREVIEW:")
    out_lines.append(source[:500] + ("..." if len(source) > 500 else ""))
    
    if outputs:
        for o_idx, out in enumerate(outputs):
            out_type = out.get('output_type', '')
            if out_type == 'stream':
                text = "".join(out.get('text', []))
                out_lines.append(f"  [Stream Output]:")
                out_lines.append(text[:2000])
            elif out_type in ['execute_result', 'display_data']:
                data = out.get('data', {})
                for mime, val in data.items():
                    if mime == 'text/plain':
                        val_str = "".join(val)
                        out_lines.append(f"  [Plain Text Output]:")
                        out_lines.append(val_str[:2000])
                    elif mime == 'text/html':
                        val_str = "".join(val)
                        out_lines.append(f"  [HTML Output (length {len(val_str)})]")

with open(r"d:\Unknown\Papper\Code\scratch\scan_dtw_output.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))
print("Successfully wrote DTW scan results to file!")
