"""Audit committed evidence, source provenance, and the compiled paper."""
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with (ROOT/path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    metadata=json.loads((ROOT/'results/fresh/metadata.json').read_text())
    assert metadata['source_sha256']==hashlib.sha256((ROOT/'src/main.cpp').read_bytes()).hexdigest(), 'Source differs from experiment executable provenance'
    audit=json.loads((ROOT/'results/derived/artifact_audit.json').read_text())
    for name,digest in audit['hashes'].items():
        assert hashlib.sha256((ROOT/'results/supplied'/name).read_bytes()).hexdigest()==digest, name
    epochs=read('results/fresh/epoch_curves.csv'); summary=read('results/fresh/comparison.csv')
    assert len(epochs)==120 and len(summary)==24
    for row in summary:
        name=f"{row['configuration']}_tau{row['tau_max']}_window{row['max_in_flight']}_run{row['run']}"
        raw=read('results/fresh/raw/'+name+'.csv')
        assert len(raw)==5
        assert abs(float(raw[-1]['elapsed_training_seconds'])-float(row['total_training_seconds']))<1e-10
        assert abs(float(raw[-1]['test_accuracy_percent'])-float(row['final_test_accuracy_percent']))<1e-10
        assert sum(int(r['accepted']) for r in raw)==int(row['accepted'])
        assert int(row['accepted'])+int(row['discarded'])==25000
        for r in raw:
            assert int(r['max_accepted_age'])<=int(row['tau_max'])
            assert int(r['max_age'])<=int(row['max_in_flight'])-1
        reached=next((r for r in raw if float(r['test_accuracy_percent'])>=90),None)
        assert (reached is None)==(row['time_to_90_percent_seconds']=='')
        if reached:
            assert abs(float(reached['elapsed_training_seconds'])-float(row['time_to_90_percent_seconds']))<1e-10
    for repeat in ['1','2','3']:
        a=[r for r in epochs if r['configuration']=='sequential' and r['run']==repeat]
        b=[r for r in epochs if r['configuration']=='synchronous' and r['run']==repeat]
        for x,y in zip(a,b):
            for metric in ['training_loss','test_loss','test_accuracy_percent','parameter_checksum']:
                assert x[metric]==y[metric]
    paper=ROOT/'report/main.pdf'
    pdf=PdfReader(paper)
    assert len(pdf.pages)<=15, 'Report exceeds page limit'
    text='\n'.join(page.extract_text() for page in pdf.pages)
    for author in ['Kaustubh Upadhya','Dhruv Pekhale','Jeet Mahajan','Tejas Tandon']:
        assert author in text, author
    assert 'https://github.com/kaustubh-dot/delayed-gradient-cpu-mlp' in text.replace('\n','')
    assert '\ufffd' not in text
    source=(ROOT/'report/main.tex').read_text(encoding='utf-8')
    assert not any(dash in source for dash in ['\u2014','\u2013'])
    assert not re.search(r'\b(TODO|TBD|PLACEHOLDER)\b',source)
    bib=(ROOT/'report/references.bib').read_text()
    keys=set(re.findall(r'@\w+\{([^,]+)',bib))
    for cite in re.findall(r'\\cite\{([^}]+)\}',source):
        assert set(cite.split(','))<=keys
    assert len(keys)==13
    for extension in ['pdf','svg','png']:
        assert len(list((ROOT/'figures').glob('*.'+extension)))==8
    assert not re.search(r'[A-Z]:[\\/]',(ROOT/'results/fresh/metadata.json').read_text()), 'Machine paths should be redacted in public metadata'
    results={'checks':'source provenance, supplied hashes, 24 raw runs / 120 epochs, targets, counts, ages, strict quality equivalence, PDF authors/page budget/repository, bibliography, figures, path redaction', 'pdf_pages':len(pdf.pages),'primary_references':len(keys),'fresh_runs':len(summary),'fresh_epochs':len(epochs),'figure_sets':8}
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
