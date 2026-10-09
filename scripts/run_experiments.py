"""Run a serial, randomized sweep and retain complete measurements."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def parse_log(text):
    lines = text.splitlines()
    header = 'epoch,average_training_loss,test_loss,test_accuracy_percent,elapsed_training_seconds'
    start = lines.index(header)
    rows = []
    for line in lines[start + 1:]:
        if not line or not line.split(',')[0].isdigit():
            break
        rows.append(dict(zip(header.split(','), line.split(','))))
    fields = {}
    for line in lines:
        if ': ' in line:
            name, value = line.split(': ', 1)
            fields[name] = value
    return rows, fields

def number(fields, name):
    return float(fields[name].split()[0].removesuffix('%'))

def cpu_name():
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    if Path('/proc/cpuinfo').exists():
        return next(line.split(':', 1)[1].strip() for line in
                    Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))
    return platform.processor()

def run(args):
    target = ROOT / args.output
    if target.exists() and any(target.iterdir()):
        raise ValueError('Choose a new output directory; existing measurements are preserved.')
    (target / 'raw').mkdir(parents=True, exist_ok=True)
    configs = [('sequential', 0, 1), ('synchronous', 0, 1)] + [
        ('delayed', tau, 8) for tau in [0, 1, 2, 4, 8, 16]]
    rng = random.Random(20261009)
    epochs, summaries, commands = [], [], []
    for repetition in range(1, args.repeats + 1):
        order = configs.copy()
        rng.shuffle(order)
        for mode, tau, window in order:
            name = f'{mode}_tau{tau}_window{window}_run{repetition}'
            exe = args.sequential if mode == 'sequential' else args.workers
            arguments = [str(args.epochs)] if mode == 'sequential' else [
                str(tau), str(window), str(args.epochs)]
            result = subprocess.run([str(exe.resolve()), *arguments], cwd=ROOT,
                                    capture_output=True, text=True, timeout=300)
            (target / 'raw' / (name + '.txt')).write_text(result.stdout, encoding='utf-8')
            if result.returncode or result.stderr:
                (target / 'raw' / (name + '.stderr.txt')).write_text(result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(result.stderr)
            rows, fields = parse_log(result.stdout)
            assert [int(row['epoch']) for row in rows] == list(range(1, args.epochs + 1))
            total = 5000 * args.epochs
            accepted = int(number(fields, 'Hidden-layer updates' if mode == 'sequential'
                                  else 'Worker 0 accepted updates'))
            rejected = 0 if mode == 'sequential' else int(number(fields, 'Worker 0 discarded gradients'))
            output = int(number(fields, 'Output-layer updates' if mode == 'sequential' else 'Worker 1 updates'))
            assert accepted + rejected == output == total
            for row in rows:
                epochs.append(dict(configuration=mode, tau_max='NA' if mode == 'sequential' else tau,
                                   max_in_flight='NA' if mode == 'sequential' else window,
                                   run=repetition, **row))
            hit = next((row for row in rows if float(row['test_accuracy_percent']) >= 90), None)
            summary = dict(run=repetition, configuration=mode,
                tau_max='NA' if mode == 'sequential' else tau,
                max_in_flight='NA' if mode == 'sequential' else window,
                hidden_update_acceptance_percent=100 * accepted / total,
                final_test_accuracy_percent=float(rows[-1]['test_accuracy_percent']),
                final_test_loss=float(rows[-1]['test_loss']),
                total_training_seconds=float(rows[-1]['elapsed_training_seconds']),
                epoch_first_at_or_above_90_percent=hit['epoch'] if hit else 'NA',
                time_to_90_percent_seconds=hit['elapsed_training_seconds'] if hit else 'NA',
                accepted=accepted, discarded=rejected, output_updates=output,
                mean_age=0 if mode == 'sequential' else number(fields, 'Average age of all gradients'),
                mean_accepted_age=0 if mode == 'sequential' else number(fields, 'Average age of accepted gradients'),
                max_age=0 if mode == 'sequential' else number(fields, 'Maximum observed gradient age'))
            for worker in [0, 1]:
                for suffix, label in [('active_seconds', 'active-path time (processing and queue pushes)'),
                                      ('pop_seconds', 'input-queue wait time')]:
                    summary[f'w{worker}_{suffix}'] = '' if mode == 'sequential' else number(
                        fields, f'Worker {worker} {label}')
            assert summary['max_age'] <= window - 1
            if mode == 'synchronous':
                assert accepted == total and summary['max_age'] == 0
            summaries.append(summary)
            commands.append(dict(run=name, executable=exe.name, arguments=arguments))
            print(name, f"{summary['total_training_seconds']:.3f}s",
                  f"accuracy={summary['final_test_accuracy_percent']:.1f}%",
                  f"acceptance={summary['hidden_update_acceptance_percent']:.2f}%", flush=True)
    write_csv(target / 'epoch_curves.csv', epochs)
    write_csv(target / 'comparison.csv', summaries)
    metadata = dict(captured_utc=datetime.now(timezone.utc).isoformat(), cpu=cpu_name(),
        logical_processors=os.cpu_count(), os=platform.platform(),
        compiler=subprocess.check_output([str(args.compiler), '--version'], text=True).splitlines()[0],
        build_flags=args.build_flags,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((ROOT / 'src').iterdir())},
        binary_sha256={exe.name: hashlib.sha256(exe.read_bytes()).hexdigest()
                       for exe in [args.sequential, args.workers]},
        dataset_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / 'data').glob('*.idx*-ubyte'))},
        subset='shuffle all training indices and select 5000; then shuffle all test indices and select 1000, using one persistent mt19937(12345); reshuffle selected training indices each epoch',
        seed=12345, learning_rate=0.01, epochs=args.epochs, repetitions=args.repeats,
        order_seed=20261009,
        timing='sum of epoch training windows; excludes loading, initialization, evaluation, shuffling and final shutdown',
        worker_pop_timing='complete input pop call across worker lifetime; includes lock, condition-variable wait and removal; can include evaluation gaps; not CPU utilization',
        affinity='not set by the experiment', background_load='not controlled or measured',
        commands=commands)
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ext = '.exe' if os.name == 'nt' else ''
    parser.add_argument('--sequential', type=Path, default=ROOT / 'build' / ('sequential_baseline' + ext))
    parser.add_argument('--workers', type=Path, default=ROOT / 'build' / ('mnist_reader' + ext))
    parser.add_argument('--compiler', type=Path, default=Path('g++'))
    parser.add_argument('--build-flags', default='not recorded')
    parser.add_argument('--output', default='results/reproduction')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.epochs <= 20:
        parser.error('repeats must be positive; epochs must be between 1 and 20')
    run(args)
