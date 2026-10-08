"""Validate the supplied workbook/CSVs and create explicitly derived timing data."""
import csv
import hashlib
import json
from pathlib import Path
import re
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/supplied'
OUT = ROOT / 'results/derived'

def read(name):
    with (RAW / name).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def main():
    OUT.mkdir(exist_ok=True)
    curves, comparison = read('epoch_curves.csv'), read('epoch_comparison.csv')
    assert len(curves) == 45 and len(comparison) == 9
    for row in comparison:
        run = [r for r in curves if (r['configuration'], r['tau_max'], r['run']) == (row['configuration'], row['tau_max'], row['run'])]
        assert [int(r['epoch']) for r in run] == [1, 2, 3, 4, 5]
        final = run[-1]
        for a, b in [('final_test_accuracy_percent', 'test_accuracy_percent'), ('final_test_loss', 'test_loss'), ('total_training_seconds', 'elapsed_training_seconds')]:
            assert abs(float(row[a]) - float(final[b])) < 1e-6
        hit = next((r for r in run if float(r['test_accuracy_percent']) >= 90), None)
        assert (hit is None) == (row['time_to_90_percent_seconds'] == 'NA')
        if hit:
            assert abs(float(row['time_to_90_percent_seconds']) - float(hit['elapsed_training_seconds'])) < 1e-6
    workbook = openpyxl.load_workbook(RAW / 'curves.xlsx', data_only=True)
    sheet = list(workbook['epoch_curves'].values)
    for values, row in zip(sheet[1:], curves):
        for value, expected in zip(values, row.values()):
            try:
                assert abs(float(value)-float(expected)) < 1e-6
            except ValueError:
                assert str(value) == expected
    assert len(sheet) == 46
    text = (RAW / 'worker_timing_runs.txt').read_text(encoding='utf-8')
    # Construct corrected headings from numeric block contents, not replacement order.
    blocks = re.split(r'(?m)^(?:Synchronous|Delayed) \([^\n]+\)\s*\n', text)[1:]
    clean, timing, epochs = [], [], []
    for block in blocks:
        def number(label):
            return float(re.search(re.escape(label) + r': ([\d.]+)', block).group(1))
        tau, window = int(number('Maximum gradient age')), int(number('Maximum batches in flight'))
        mode = 'synchronous' if window == 1 else 'delayed'
        clean.append(f'{mode.capitalize()} ({tau} {window} 5)\n\n' + block.strip())
        row = dict(configuration=mode, tau_max=tau, max_in_flight=window, total_training_seconds=number('Total training time'), accepted=number('Worker 0 accepted updates'), discarded=number('Worker 0 discarded gradients'), hidden_update_acceptance_percent=number('Worker 0 update acceptance'), final_test_accuracy_percent=number('Test accuracy after training'), mean_age=number('Average age of all gradients'), mean_accepted_age=number('Average age of accepted gradients'), max_age=number('Maximum observed gradient age'))
        for worker in (0, 1):
            row[f'w{worker}_active_seconds'] = number(f'Worker {worker} active-path time (processing and queue pushes)')
            row[f'w{worker}_pop_seconds'] = number(f'Worker {worker} input-queue wait time')
        timing.append(row)
        lines = block.strip().splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith('epoch,'))
        for r in csv.DictReader(lines[header:header+6]):
            epochs.append(dict(configuration=mode, tau_max=tau, max_in_flight=window, run='instrumented', **r))
    (OUT / 'worker_timing_corrected.txt').write_text('\n\n'.join(clean)+'\n', encoding='utf-8')
    for name, rows in [('instrumented_comparison.csv',timing),('instrumented_epochs.csv',epochs)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest = dict(checks=['45 epoch rows / nine runs', 'all final metrics and time-to-target match CSV checkpoints', 'workbook epoch sheet matches CSV to 1e-6', 'two timing runs identified using numeric settings'], hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.iterdir()}, workbook_sheets=workbook.sheetnames, caveats=['Original compiler, CPU, queue capacity, source, and exact subset identity unavailable', 'Timing headings are swapped in both raw timing log and original demo notes', 'Instrumented runs are separate observations, not CSV run replacements', 'tau=16 is mentioned in guides, but no supplied numeric tau=16 results are available'])
    (OUT/'artifact_audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Supplied artifact audit passed; derived timing files written')

if __name__ == '__main__':
    main()
