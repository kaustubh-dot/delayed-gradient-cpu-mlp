"""Verify recorded checkpoints and raw experiment summaries."""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from run_experiments import parse_log, number, write_csv

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with (ROOT / path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def close(actual, expected, tolerance=1.1e-6):
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)

def extract_worker_timing():
    text = (ROOT / 'results/recorded/worker_timing_runs.txt').read_text()
    chunks = text.split('epoch,average_training_loss,test_loss,test_accuracy_percent,elapsed_training_seconds')[1:]
    rows = []
    for chunk in chunks:
        _, fields = parse_log('epoch,average_training_loss,test_loss,test_accuracy_percent,elapsed_training_seconds' + chunk)
        tau = int(number(fields, 'Maximum gradient age'))
        window = int(number(fields, 'Maximum batches in flight'))
        row = dict(configuration='synchronous' if (tau, window) == (0, 1) else 'delayed',
                   tau_max=tau, max_in_flight=window,
                   total_training_seconds=number(fields, 'Total training time'))
        for worker in [0, 1]:
            row[f'w{worker}_active_seconds'] = number(fields, f'Worker {worker} active-path time (processing and queue pushes)')
            row[f'w{worker}_pop_seconds'] = number(fields, f'Worker {worker} input-queue wait time')
        rows.append(row)
    assert {(r['tau_max'], r['max_in_flight']) for r in rows} == {(4, 8), (0, 1)}
    (ROOT / 'results/summary').mkdir(exist_ok=True)
    write_csv(ROOT / 'results/summary/instrumented_comparison.csv', rows)

def check_checkpoints(folder, expected_runs):
    epochs = read(f'results/{folder}/epoch_curves.csv')
    comparisons = read(f'results/{folder}/comparison.csv')
    assert len(comparisons) == expected_runs and len(epochs) == 5 * expected_runs
    grouped = defaultdict(list)
    for row in epochs:
        grouped[row['configuration'], row['tau_max'], row['run']].append(row)
    for row in comparisons:
        checkpoints = grouped[row['configuration'], row['tau_max'], row['run']]
        assert [int(r['epoch']) for r in checkpoints] == [1, 2, 3, 4, 5]
        for field, target in [('test_accuracy_percent', 'final_test_accuracy_percent'),
                              ('test_loss', 'final_test_loss'),
                              ('elapsed_training_seconds', 'total_training_seconds')]:
            close(checkpoints[-1][field], row[target])
        hit = next((r for r in checkpoints if float(r['test_accuracy_percent']) >= 90), None)
        if hit:
            close(row['epoch_first_at_or_above_90_percent'], hit['epoch'])
            close(row['time_to_90_percent_seconds'], hit['elapsed_training_seconds'])
        else:
            assert row['time_to_90_percent_seconds'] == row['epoch_first_at_or_above_90_percent'] == 'NA'
    return comparisons, grouped

def check_reproduction():
    comparisons, grouped = check_checkpoints('reproduction', 24)
    for row in comparisons:
        mode, tau, repeat = row['configuration'], row['tau_max'], row['run']
        window = 1 if mode in ['sequential', 'synchronous'] else 8
        name = f"{mode}_tau{0 if tau == 'NA' else tau}_window{window}_run{repeat}"
        raw, fields = parse_log((ROOT / 'results/reproduction/raw' / (name + '.txt')).read_text())
        for stored, original in zip(grouped[mode, tau, repeat], raw):
            for field in original:
                close(stored[field], original[field])
        assert len(raw) == 5
        hidden = number(fields, 'Hidden-layer updates' if mode == 'sequential' else 'Worker 0 accepted updates')
        discarded = 0 if mode == 'sequential' else number(fields, 'Worker 0 discarded gradients')
        output = number(fields, 'Output-layer updates' if mode == 'sequential' else 'Worker 1 updates')
        assert hidden + discarded == output == 25000
        close(row['accepted'], hidden)
        close(row['discarded'], discarded)
        close(row['output_updates'], output)
        close(row['hidden_update_acceptance_percent'], 100 * hidden / 25000)
        close(row['total_training_seconds'], number(fields, 'Total training time'))
        for field, label in [('mean_age', 'Average age of all gradients'),
                             ('mean_accepted_age', 'Average age of accepted gradients'),
                             ('max_age', 'Maximum observed gradient age')]:
            value = 0 if mode == 'sequential' else number(fields, label)
            close(row[field], value)
        assert float(row['max_age']) <= window - 1
        if mode != 'sequential':
            assert float(row['mean_accepted_age']) <= int(tau)
            for worker in [0, 1]:
                for suffix, label in [('active_seconds', 'active-path time (processing and queue pushes)'),
                                      ('pop_seconds', 'input-queue wait time')]:
                    close(row[f'w{worker}_{suffix}'], number(fields, f'Worker {worker} {label}'))
        if mode == 'synchronous' or (mode == 'delayed' and int(tau) >= 8):
            assert hidden == 25000 and discarded == 0
    for repetition in ['1', '2', '3']:
        sequential = grouped['sequential', 'NA', repetition]
        synchronous = grouped['synchronous', '0', repetition]
        for seq, sync in zip(sequential, synchronous):
            for field in ['average_training_loss', 'test_loss', 'test_accuracy_percent']:
                assert seq[field] == sync[field]
    metadata = json.loads((ROOT / 'results/reproduction/metadata.json').read_text())
    for name, digest in metadata['source_sha256'].items():
        assert hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() == digest, name

def check_pilot():
    delayed = read('results/pilot/delayed_repetitions.csv')
    interleaved = read('results/pilot/interleaved_comparison.csv')
    assert len(delayed) == 30 and len(interleaved) == 15
    for row in delayed:
        assert int(row['worker0_updates']) + int(row['worker0_discarded']) == int(row['worker1_updates']) == 5000
        close(row['worker0_acceptance'].removesuffix('%'), int(row['worker0_updates']) / 50, 0.0001)
        assert float(row['max_gradient_age']) <= 7
        if int(row['tau_max']) >= 8:
            assert int(row['worker0_discarded']) == 0
    assert all(sum(r['tau_max'] == str(t) for r in delayed) == 5 for t in [0, 1, 2, 4, 8, 16])
    assert all(sum(r['mode'] == m for r in interleaved) == 5 for m in ['sequential', 'synchronous', 'delayed_tau8'])

if __name__ == '__main__':
    check_checkpoints('recorded', 9)
    check_reproduction()
    check_pilot()
    print('Recorded checkpoints, 24 raw reproduction runs, source hashes and pilot counts verified.')
