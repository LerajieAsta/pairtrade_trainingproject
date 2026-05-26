import json

file_path = r"d:\Unknown\Papper\Code\Ref_CODE\純SSD配對交易-[版本1].ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    cell_type = cell.get('cell_type', '')
    print(f"\n--- Cell {idx} ({cell_type}) ---")
    source = "".join(cell.get('source', []))
    print("CODE/TEXT PREVIEW:")
    print(source[:200] + ("..." if len(source) > 200 else ""))
    
    outputs = cell.get('outputs', [])
    if outputs:
        print(f"Outputs count: {len(outputs)}")
        for o_idx, out in enumerate(outputs):
            out_type = out.get('output_type', '')
            print(f"  Output {o_idx} type: {out_type}")
            if out_type == 'stream':
                text = "".join(out.get('text', []))
                print(f"    Stream Text (first 200 chars): {text[:200]}...")
            elif out_type == 'execute_result' or out_type == 'display_data':
                data = out.get('data', {})
                for mime, val in data.items():
                    if mime == 'text/plain':
                        val_str = "".join(val)
                        print(f"    Plain text (first 200 chars): {val_str[:200]}...")
                    elif mime == 'text/html':
                        val_str = "".join(val)
                        print(f"    HTML (length {len(val_str)})")
