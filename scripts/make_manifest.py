"""Hash the Git-index submission file set, excluding this manifest's own hash."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'docs/submission_manifest.json'
files=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files if name and ROOT/name != TARGET}
TARGET.write_text(json.dumps({'algorithm':'SHA-256','scope':'Git-index files, excluding the manifest itself','files':hashes},indent=2)+'\n',encoding='utf-8')
print('Submission files hashed:',len(hashes))
