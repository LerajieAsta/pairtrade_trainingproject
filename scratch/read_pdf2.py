import pdfplumber
import sys

pdf_path = r'C:\Clark\YZU\Papper\Code\ref\2021-Pairs Trading via Unsupervised Learning.pdf'

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            print(f"\n{'='*60}")
            print(f"PAGE {i+1}")
            print('='*60)
            print(text.encode('ascii', 'replace').decode('ascii'))
