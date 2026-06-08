import sys
import struct

def extract_pdf_text(filepath):
    """Simple PDF text extractor without external dependencies."""
    results = []
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Try to extract text streams
        import re
        # Find all text between BT and ET markers
        text_blocks = re.findall(b'BT(.+?)ET', content, re.DOTALL)
        
        for block in text_blocks:
            # Extract text from Tj and TJ operators
            texts = re.findall(b'\(([^)]*)\)\s*Tj', block)
            for t in texts:
                try:
                    results.append(t.decode('latin-1', errors='replace'))
                except:
                    pass
            
            # TJ arrays
            tj_arrays = re.findall(b'\[([^\]]+)\]\s*TJ', block)
            for arr in tj_arrays:
                parts = re.findall(b'\(([^)]*)\)', arr)
                for p in parts:
                    try:
                        results.append(p.decode('latin-1', errors='replace'))
                    except:
                        pass
    except Exception as e:
        return f"Error: {e}"
    
    return '\n'.join(results)

print(extract_pdf_text(r'C:\Clark\YZU\Papper\Code\ref\2021-Pairs Trading via Unsupervised Learning.pdf'))
