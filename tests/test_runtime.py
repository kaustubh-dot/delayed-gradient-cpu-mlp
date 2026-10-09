"""Integration checks on generated IDX fixtures; no MNIST download required."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_experiments import parse_log, number

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ext = '.exe' if os.name == 'nt' else ''
        cls.sequential = ROOT / 'build' / ('sequential_baseline' + ext)
        cls.workers = ROOT / 'build' / ('mnist_reader' + ext)
        cls.temp = tempfile.TemporaryDirectory(dir=ROOT / 'build')
        cls.cwd = Path(cls.temp.name)
        data = cls.cwd / 'data'
        data.mkdir()
        for prefix, count in [('train', 5000), ('t10k', 1000)]:
            pixels = bytearray(count * 784)
            for i in range(count):
                label = i % 10
                start = i * 784 + label * 60
                pixels[start:start + 50] = bytes([255]) * 50
            (data / (prefix + '-images.idx3-ubyte')).write_bytes(
                struct.pack('>IIII', 2051, count, 28, 28) + pixels)
            (data / (prefix + '-labels.idx1-ubyte')).write_bytes(
                struct.pack('>II', 2049, count) + bytes(i % 10 for i in range(count)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, exe, arguments, success=True):
        result = subprocess.run([str(exe), *map(str, arguments)], cwd=self.cwd,
                                text=True, capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            rows, fields = parse_log(result.stdout)
            self.assertEqual([int(row['epoch']) for row in rows], [1, 2])
            return rows, fields
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('Error:', result.stderr)

    def test_synchronous_matches_sequential(self):
        sequential, sf = self.execute(self.sequential, [2])
        synchronous, wf = self.execute(self.workers, [0, 1, 2])
        for s, w in zip(sequential, synchronous):
            for metric in ['average_training_loss', 'test_loss', 'test_accuracy_percent']:
                self.assertEqual(s[metric], w[metric])
        self.assertEqual(number(sf, 'Hidden-layer updates'), 10000)
        self.assertEqual(number(wf, 'Worker 0 accepted updates'), 10000)
        self.assertEqual(number(wf, 'Maximum observed gradient age'), 0)

    def test_delayed_counts_and_age(self):
        for tau in [0, 4, 8, 16]:
            _, fields = self.execute(self.workers, [tau, 8, 2])
            accepted = number(fields, 'Worker 0 accepted updates')
            discarded = number(fields, 'Worker 0 discarded gradients')
            self.assertEqual(accepted + discarded, 10000)
            self.assertEqual(number(fields, 'Worker 1 updates'), 10000)
            self.assertLessEqual(number(fields, 'Maximum observed gradient age'), 7)
            self.assertLessEqual(number(fields, 'Average age of accepted gradients'), tau)
            if tau >= 8:
                self.assertEqual(discarded, 0)

    def test_invalid_configuration(self):
        for arguments in [[3, 8, 2], [4, 3, 2], [4, 8, 0], [4, 8, 21]]:
            self.execute(self.workers, arguments, success=False)
        for arguments in [[0], [-1], [21]]:
            self.execute(self.sequential, arguments, success=False)

    def test_worker_error_joins_threads(self):
        labels = self.cwd / 'data/train-labels.idx1-ubyte'
        original = labels.read_bytes()
        try:
            labels.write_bytes(original[:8] + bytes([10]) * 5000)
            self.execute(self.workers, [4, 8, 2], success=False)
        finally:
            labels.write_bytes(original)

if __name__ == '__main__':
    unittest.main()
