"""Download canonical MNIST IDX files; verify published compressed-file MD5s."""
import argparse
import gzip
import hashlib
from pathlib import Path
import urllib.request
import time

FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}

def download(target):
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        packed = target / name
        if not packed.exists() or hashlib.md5(packed.read_bytes()).hexdigest() != expected:
            mirrors = ['https://storage.googleapis.com/cvdf-datasets/mnist/', 'https://ossci-datasets.s3.amazonaws.com/mnist/']
            last_error = None
            for mirror in mirrors:
                for attempt in range(3):
                    try:
                        partial = target / (name + '.part')
                        with urllib.request.urlopen(mirror + name, timeout=30) as response, partial.open('wb') as output:
                            while chunk := response.read(256 * 1024):
                                output.write(chunk)
                        if hashlib.md5(partial.read_bytes()).hexdigest() != expected:
                            raise ValueError('Compressed-file checksum mismatch')
                        partial.replace(packed)
                        break
                    except (OSError, ValueError) as error:
                        last_error = error
                        print('Retry', name, type(error).__name__, flush=True)
                        time.sleep(1)
                else:
                    continue
                break
            else:
                raise RuntimeError('All MNIST mirrors failed') from last_error
        if hashlib.md5(packed.read_bytes()).hexdigest() != expected:
            raise ValueError('MNIST checksum mismatch: ' + name)
        unpacked = target / name.removesuffix('.gz').replace('-idx', '.idx')
        unpacked.write_bytes(gzip.decompress(packed.read_bytes()))
        print(name, 'verified', hashlib.sha256(unpacked.read_bytes()).hexdigest(), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data'))
    download(p.parse_args().data)
