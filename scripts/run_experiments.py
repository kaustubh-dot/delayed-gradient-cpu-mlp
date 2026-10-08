"""Repeat a randomized serial sweep and preserve every raw stdout/stderr."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def cpu_name():
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    if Path('/proc/cpuinfo').exists():
        return next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    return platform.processor()

def main(args):
    target = ROOT / args.output
    if target.exists() and any(target.iterdir()):
        raise ValueError('Refusing to overwrite experiment output; choose a fresh --output directory')
    (target / 'raw').mkdir(parents=True, exist_ok=True)
    configs = [('sequential', 0, 1), ('synchronous', 0, 1)] + [('delayed', t, 8) for t in (0, 1, 2, 4, 8, 16)]
    order_rng = random.Random(20261008)
    all_epochs, summaries, commands = [], [], []
    for repeat in range(1, args.repeats + 1):
        order = configs.copy()
        order_rng.shuffle(order)
        for mode, tau, window in order:
            name = f'{mode}_tau{tau}_window{window}_run{repeat}'
            command = [str(args.exe.resolve()), '--mode', 'sequential' if mode == 'sequential' else 'workers', '--tau', str(tau), '--in-flight', str(window), '--epochs', str(args.epochs), '--train', '5000', '--test', '1000', '--seed', '12345', '--lr', '0.01', '--data', str(args.data)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            (target / 'raw' / (name + '.csv')).write_text(result.stdout, encoding='utf-8')
            (target / 'raw' / (name + '.txt')).write_text(result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(result.stderr)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            if len(rows) != args.epochs:
                raise ValueError('Incomplete epoch log: ' + name)
            for row in rows:
                assert int(row['accepted']) + int(row['discarded']) == 5000
                assert int(row['output_updates']) == 5000
                assert int(row['max_accepted_age']) <= tau
                assert int(row['max_age']) <= window - 1
                all_epochs.append(dict(configuration=mode, tau_max=tau, max_in_flight=window, run=repeat, **row))
            final = rows[-1]
            accepted = sum(int(r['accepted']) for r in rows)
            rejected = sum(int(r['discarded']) for r in rows)
            hit = next((r for r in rows if float(r['test_accuracy_percent']) >= 90), None)
            elapsed = float(final['elapsed_training_seconds'])
            summary = dict(configuration=mode, tau_max=tau, max_in_flight=window, run=repeat,
                final_test_accuracy_percent=float(final['test_accuracy_percent']), final_test_loss=float(final['test_loss']),
                total_training_seconds=elapsed, accepted=accepted, discarded=rejected, hidden_update_acceptance_percent=accepted/(5000*args.epochs)*100,
                completed_examples=5000*args.epochs, completed_example_rate=5000*args.epochs/elapsed, accepted_update_rate=accepted/elapsed,
                epoch_first_at_or_above_90_percent=int(hit['epoch']) if hit else '', time_to_90_percent_seconds=float(hit['elapsed_training_seconds']) if hit else '',
                mean_age=sum(float(r['mean_age']) for r in rows)/len(rows), max_age=max(int(r['max_age']) for r in rows),
                max_accepted_age=max(int(r['max_accepted_age']) for r in rows),
                mean_accepted_age=sum(float(r['mean_accepted_age'])*int(r['accepted']) for r in rows)/accepted if accepted else 0)
            for key, value in re.findall(r'([a-z0-9_]+)=([-+\d.eE]+)', result.stderr):
                summary[key] = float(value)
            summaries.append(summary)
            commands.append(dict(name=name, command=['build/mlp'] + command[1:-1] + ['data/mnist']))
            print(name, f"{elapsed:.3f}s", f"accuracy={summary['final_test_accuracy_percent']:.1f}%", f"acceptance={summary['hidden_update_acceptance_percent']:.2f}%", flush=True)
    write_csv(target / 'epoch_curves.csv', all_epochs)
    write_csv(target / 'comparison.csv', summaries)
    metadata = dict(captured_utc=datetime.now(timezone.utc).isoformat(), cpu=cpu_name(), logical_processors=os.cpu_count(), os=platform.platform(),
        compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        build_flags='-O2 -std=c++17 -pthread -Wall -Wextra -Wpedantic',
        source_sha256=hashlib.sha256((ROOT / 'src/main.cpp').read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        dataset_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.iterdir() if not p.name.endswith('.gz')},
        selected_subset='first 5000 train / first 1000 official test, shuffle train each epoch',
        seed=12345, learning_rate=0.01, epochs=args.epochs, repetitions=args.repeats,
        order_seed=20261008, timing='steady_clock; sum of epoch training windows; excludes evaluation and startup',
        pop_timing='mutex acquisition and predicate wait, before deque removal; accumulated across worker lifetime and can include evaluation gaps; not CPU utilization',
        affinity='not pinned', background_load='not controlled or measured', commands=commands)
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=ROOT / 'build' / ('mlp.exe' if os.name == 'nt' else 'mlp'))
    p.add_argument('--data', type=Path, default=ROOT / 'data/mnist')
    p.add_argument('--output', default='results/fresh')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--epochs', type=int, default=5)
    main(p.parse_args())
