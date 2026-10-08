"""Independent second-pass checks, using raw logs rather than plot summaries."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--originals', type=Path, help='Optional folder containing the original supplied files')
p.add_argument('--experiment-binary', type=Path, help='Optional retained binary from the original reference sweep, not a rebuild')
p.add_argument('--write', action='store_true', help='Write the computed evidence JSON; otherwise only verify')
args = p.parse_args()

def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def near(a, b, tolerance=1e-9):
    assert abs(float(a) - float(b)) <= tolerance, (a, b)

summary = rows(ROOT / 'results/fresh/comparison.csv')
combined = rows(ROOT / 'results/fresh/epoch_curves.csv')
assert len(summary) == 24 and len(combined) == 120
observed = defaultdict(list)
for run in summary:
    stem = f"{run['configuration']}_tau{run['tau_max']}_window{run['max_in_flight']}_run{run['run']}"
    raw = rows(ROOT / f'results/fresh/raw/{stem}.csv')
    assert [int(r['epoch']) for r in raw] == list(range(1, 6))
    group = [r for r in combined if all(r[k] == run[k] for k in ('configuration','tau_max','max_in_flight','run'))]
    assert len(group) == 5
    for a,b in zip(raw, group):
        assert all(a[k] == b[k] for k in a)
        assert int(a['accepted']) + int(a['discarded']) == 5000
        assert int(a['output_updates']) == 5000
        assert int(a['max_accepted_age']) <= int(run['tau_max'])
        assert int(a['max_age']) <= int(run['max_in_flight']) - 1
        assert int(a['max_pending']) <= int(run['max_in_flight'])
        assert 0 <= float(a['mean_accepted_age']) <= int(a['max_accepted_age'])
        assert 0 <= float(a['mean_age']) <= int(a['max_age'])
        assert float(a['epoch_training_seconds']) > 0
    near(sum(float(r['epoch_training_seconds']) for r in raw), raw[-1]['elapsed_training_seconds'])
    near(raw[-1]['elapsed_training_seconds'], run['total_training_seconds'])
    near(raw[-1]['test_loss'], run['final_test_loss'])
    near(raw[-1]['test_accuracy_percent'], run['final_test_accuracy_percent'])
    accepted = sum(int(r['accepted']) for r in raw)
    discarded = sum(int(r['discarded']) for r in raw)
    assert accepted == int(run['accepted']) and discarded == int(run['discarded'])
    near(accepted / 25000 * 100, run['hidden_update_acceptance_percent'])
    near(25000 / float(run['total_training_seconds']), run['completed_example_rate'])
    near(accepted / float(run['total_training_seconds']), run['accepted_update_rate'])
    near(mean(float(r['mean_age']) for r in raw), run['mean_age'])
    near(sum(float(r['mean_accepted_age']) * int(r['accepted']) for r in raw) / accepted, run['mean_accepted_age'])
    assert max(int(r['max_age']) for r in raw) == int(run['max_age'])
    hit = next((r for r in raw if float(r['test_accuracy_percent']) >= 90), None)
    if hit:
        near(hit['elapsed_training_seconds'], run['time_to_90_percent_seconds'])
        assert hit['epoch'] == run['epoch_first_at_or_above_90_percent']
    else:
        assert run['time_to_90_percent_seconds'] == run['epoch_first_at_or_above_90_percent'] == ''
    stderr = (ROOT / f'results/fresh/raw/{stem}.txt').read_text()
    for key, value in re.findall(r'([a-z0-9_]+)=([-+\d.eE]+)', stderr):
        near(value, run[key])
    assert int(float(run['queue_capacity'])) == 2 * int(run['max_in_flight'])
    for channel in ('hidden_inbox_max','activation_queue_max','result_queue_max'):
        assert float(run[channel]) <= int(run['max_in_flight'])
    observed[(run['configuration'], run['tau_max'])].append(run)

old_epochs = rows(ROOT / 'results/supplied/epoch_curves.csv')
old_final = rows(ROOT / 'results/supplied/epoch_comparison.csv')
assert len(old_epochs) == 45 and len(old_final) == 9
if args.originals:
    for directory in ('results/supplied','docs/supplied'):
        for file in (ROOT/directory).iterdir():
            original=args.originals/file.name
            assert original.read_bytes() == file.read_bytes(), file.name
old_sync=[r for r in old_epochs if r['configuration']=='synchronous']
for repeat in ('1','2'):
    sequential=[r for r in old_epochs if r['configuration']=='sequential' and r['run']==repeat]
    for a,b in zip(sequential,old_sync):
        for field in ('average_training_loss','test_loss','test_accuracy_percent'):
            near(a[field],b[field])
old_groups = defaultdict(list)
for run in old_final:
    group = [r for r in old_epochs if all(r[k] == run[k] for k in ('configuration','tau_max','run'))]
    assert [int(r['epoch']) for r in group] == list(range(1,6))
    for a,b in [('final_test_loss','test_loss'),('final_test_accuracy_percent','test_accuracy_percent'),('total_training_seconds','elapsed_training_seconds')]:
        near(run[a], group[-1][b])
    hit = next((r for r in group if float(r['test_accuracy_percent']) >= 90), None)
    if hit:
        near(run['time_to_90_percent_seconds'], hit['elapsed_training_seconds'])
        assert run['epoch_first_at_or_above_90_percent'] == hit['epoch']
    else:
        assert run['time_to_90_percent_seconds'] == run['epoch_first_at_or_above_90_percent'] == 'NA'
    old_groups[(run['configuration'], run['tau_max'])].append(run)

workbook = openpyxl.load_workbook(ROOT/'results/supplied/curves.xlsx', data_only=True)
cells = list(workbook['epoch_curves'].values)
assert len(cells) == 46 and tuple(old_epochs[0]) == tuple(cells[0])
for line, record in zip(cells[1:], old_epochs):
    assert len(line) == len(record)
    for cell,value in zip(line,record.values()):
        try:
            near(cell,value,1e-6)
        except (ValueError, TypeError):
            assert str(cell) == value

def metrics(groups):
    out={}
    for (configuration,tau), runs in groups.items():
        out[f'{configuration}/{tau}']={
            'runs':len(runs),
            'mean_seconds':mean(float(r['total_training_seconds']) for r in runs),
            'mean_final_accuracy_percent':mean(float(r['final_test_accuracy_percent']) for r in runs),
            'mean_acceptance_percent':mean(float(r['hidden_update_acceptance_percent']) for r in runs),
            'target_runs':sum(r['time_to_90_percent_seconds'] not in ('','NA') for r in runs),
            'mean_target_seconds':mean(float(r['time_to_90_percent_seconds']) for r in runs if r['time_to_90_percent_seconds'] not in ('','NA')) if any(r['time_to_90_percent_seconds'] not in ('','NA') for r in runs) else None,
        }
    return out

fresh, old = metrics(observed), metrics(old_groups)
for setting,runs in observed.items():
    selected = [r for r in combined if (r['configuration'],r['tau_max']) == setting]
    fresh['/'.join(setting)]['pooled_accepted_age'] = sum(float(r['mean_accepted_age']) * int(r['accepted']) for r in selected) / sum(int(r['accepted']) for r in selected)
    fresh['/'.join(setting)]['mean_returned_age'] = mean(float(r['mean_age']) for r in selected)
    fresh['/'.join(setting)]['max_returned_age'] = max(int(r['max_age']) for r in selected)

# Independently compare measured means with every generated table cell.
for name, values in [('supplied',old),('fresh',fresh)]:
    table = (ROOT / f'report/{name}_table.tex').read_text()
    for key, record in values.items():
        mode,tau=key.split('/')
        label='Sequential' if mode=='sequential' else 'Sync (0,1)' if mode=='synchronous' else f'Delay {tau}'
        line = next(line for line in table.splitlines() if line.startswith(label+' &'))
        fields=[f.strip().removesuffix('\\\\').strip() for f in line.split('&')]
        assert fields == [label,str(record['runs']),f"{record['mean_seconds']:.3f}",f"{record['mean_final_accuracy_percent']:.2f}",f"{record['mean_acceptance_percent']:.2f}",f"{record['mean_target_seconds']:.3f}" if record['mean_target_seconds'] is not None else 'NR'], fields

metadata=json.loads((ROOT/'results/fresh/metadata.json').read_text())
assert metadata['source_sha256'] == hashlib.sha256((ROOT/'src/main.cpp').read_bytes()).hexdigest()
if args.experiment_binary:
    assert metadata['binary_sha256'] == hashlib.sha256(args.experiment_binary.read_bytes()).hexdigest()
data_available=(ROOT/'data/mnist').exists()
if data_available:
    for name,digest in metadata['dataset_sha256'].items():
        assert hashlib.sha256((ROOT/'data/mnist'/name).read_bytes()).hexdigest() == digest
canonical = {'train-images-idx3-ubyte.gz':'f68b3c2dcbeaaa9fbdd348bbdeb94873','train-labels-idx1-ubyte.gz':'d53e105ee54ea40749a09fcbcd1e9432','t10k-images-idx3-ubyte.gz':'9fb629c4189551a2d022fa330f9573f3','t10k-labels-idx1-ubyte.gz':'ec29112dd5afa0611ce80d1b7f02629c'}
if data_available:
    for name,digest in canonical.items():
        assert hashlib.md5((ROOT/'data/mnist'/name).read_bytes()).hexdigest() == digest

report={
    'checks_passed':['all 24 raw runs match combined and run summaries','all 120 count/age/admission invariants','all raw stderr counters match summaries','all nine original final rows and target epochs match original curves','all 45 workbook rows match CSV, including headers','original sequential/synchronous quality curves match',*(['preserved supplied evidence and contextual guides/presentation byte-match provided originals'] if args.originals else []),'all numeric cells of both report tables rederived','source hash matches metadata',*(['retained experiment binary matches metadata'] if args.experiment_binary else []),*(['four dataset hashes match metadata','four compressed MNIST files match canonical checksums'] if data_available else [])],
    'fresh':fresh,'supplied':old,
    'fresh_sync_over_delay4_ratio':fresh['synchronous/0']['mean_seconds']/fresh['delayed/4']['mean_seconds'],
    'fresh_seq_over_delay4_ratio':fresh['sequential/0']['mean_seconds']/fresh['delayed/4']['mean_seconds'],
    'supplied_seq_over_delay4_ratio':old['sequential/NA']['mean_seconds']/old['delayed/4']['mean_seconds'],
    'supplied_seq_over_delay4_target_ratio':old['sequential/NA']['mean_target_seconds']/old['delayed/4']['mean_target_seconds'],
}
if args.write:
    (ROOT/'results/derived/second_pass_numeric_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'checks_passed':report['checks_passed'],'ratios':{k:v for k,v in report.items() if k.endswith('ratio')}},indent=2))
