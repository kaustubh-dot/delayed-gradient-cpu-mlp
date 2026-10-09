"""Check the final PDF's length and required submission details."""
from pathlib import Path
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
reader = PdfReader(ROOT / 'report/main.pdf')
assert 12 <= len(reader.pages) <= 15, len(reader.pages)
text = '\n'.join(page.extract_text() for page in reader.pages)
for name in ['Kaustubh Upadhya', 'Dhruv Pekhale', 'Jeet Mahajan', 'Tejas Tandon']:
    assert name in text, name
assert 'Sri City' in text
assert 'https://github.com/kaustubh-dot/delayed-gradient-cpu-mlp' in text.replace('\n', '')
assert '\ufffd' not in text
assert not re.search(r'\[\?\]|\?\?', text)
print(f'{len(reader.pages)}-page PDF: author details, repository link and text checks passed.')
