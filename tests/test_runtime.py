"""Behavioral integration tests, using an IDX fixture rather than downloading MNIST."""
import csv
import io
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get('MLP_EXE', ROOT / 'build' / ('mlp.exe' if os.name == 'nt' else 'mlp')))

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(EXE.exists(), 'Build the runtime before executing integration tests')

    def run_case(self, mode, tau=0, window=1):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as tmp:
            d = Path(tmp)
            n = 40
            pixels = bytes((i * 7 + j * 3) % 256 for i in range(n) for j in range(784))
            labels = bytes(i % 10 for i in range(n))
            for prefix in ('train', 't10k'):
                (d / (prefix + '-images-idx3-ubyte')).write_bytes(struct.pack('>IIII', 2051, n, 28, 28) + pixels)
                (d / (prefix + '-labels-idx1-ubyte')).write_bytes(struct.pack('>II', 2049, n) + labels)
            r = subprocess.run([str(EXE), '--mode', mode, '--tau', str(tau), '--in-flight', str(window), '--epochs', '2', '--train', '40', '--test', '40', '--data', str(d)], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = list(csv.DictReader(io.StringIO(r.stdout)))
            self.assertEqual([int(row['epoch']) for row in rows], [1, 2], 'Missing or misordered checkpoint rows')
            return rows

    def test_math_and_queue_contracts(self):
        r = subprocess.run([str(EXE), '--self-test'], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('finite-difference', r.stdout)

    def test_synchronous_matches_sequential(self):
        seq = self.run_case('sequential')
        sync = self.run_case('workers')
        for a, b in zip(seq, sync):
            for key in ('training_loss', 'test_loss', 'test_accuracy_percent', 'parameter_checksum'):
                self.assertAlmostEqual(float(a[key]), float(b[key]), places=9, msg=key)
            self.assertEqual(int(b['accepted']), 40)
            self.assertEqual(int(b['discarded']), 0)
            self.assertEqual(int(b['max_age']), 0)

    def test_async_conserves_counts_and_bounds_accepted_age(self):
        for tau in (0, 1, 4, 8):
            for row in self.run_case('workers', tau, 8):
                self.assertEqual(int(row['accepted']) + int(row['discarded']), 40)
                self.assertEqual(int(row['output_updates']), 40)
                self.assertLessEqual(int(row['max_accepted_age']), tau)
                self.assertLessEqual(int(row['max_age']), 7)
                self.assertLessEqual(int(row['max_pending']), 8)

    def test_rejects_invalid_configuration(self):
        r = subprocess.run([str(EXE), '--in-flight', '0'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('positive', r.stderr)

if __name__ == '__main__':
    unittest.main()
