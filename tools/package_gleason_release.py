"""Build and verify a portable source ZIP with per-member checksums."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    files=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    files=sorted(set(f for f in files if f))
    manifest={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:
            path=root/name
            if path.is_symlink(): raise ValueError(f'Symlink not allowed in release: {name}')
            payload=path.read_bytes()
            manifest[name]=hashlib.sha256(payload).hexdigest()
            archive.writestr('dsa5206-proj/'+name,payload)
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        archive.writestr('dsa5206-proj/RELEASE.json',json.dumps(dict(base_commit=commit,files=manifest),indent=2))
    with zipfile.ZipFile(args.output) as archive:
        for name,expected in manifest.items():
            if hashlib.sha256(archive.read('dsa5206-proj/'+name)).hexdigest()!=expected:
                raise ValueError(f'Archive verification failed: {name}')
    digest=hashlib.sha256()
    with args.output.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    print(json.dumps(dict(file=str(args.output),members=len(manifest),bytes=args.output.stat().st_size,
                          sha256=digest.hexdigest()),indent=2))


if __name__=='__main__':
    main()
