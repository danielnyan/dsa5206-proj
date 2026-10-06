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
    parser.add_argument('--include', action='append', default=[], metavar='REPO_PATH',
                        help='Overlay ZIP: include only these changed paths plus refreshed RELEASE.json')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    files=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z']).decode().split('\0')
    # RELEASE.json belongs to the built archive, never to its own source list.
    # Respect tracked deletions when packaging an uncommitted working tree.
    files=sorted(set(f for f in files if f and f != 'RELEASE.json' and (root/f).exists()))
    selected = set(args.include) if args.include else set(files)
    if selected - set(files):
        raise ValueError('Overlay paths must be existing repository source files')
    manifest={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in files:
            path=root/name
            if path.is_symlink(): raise ValueError(f'Symlink not allowed in release: {name}')
            payload=path.read_bytes()
            manifest[name]=hashlib.sha256(payload).hexdigest()
            if name in selected:
                archive.writestr('dsa5206-proj/'+name,payload)
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        archive.writestr('dsa5206-proj/RELEASE.json',json.dumps(dict(base_commit=commit,files=manifest),indent=2))
    with zipfile.ZipFile(args.output) as archive:
        for name,expected in manifest.items():
            if name not in selected:
                continue
            if hashlib.sha256(archive.read('dsa5206-proj/'+name)).hexdigest()!=expected:
                raise ValueError(f'Archive verification failed: {name}')
    digest=hashlib.sha256()
    with args.output.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    print(json.dumps(dict(file=str(args.output),members=len(selected),overlay=bool(args.include),bytes=args.output.stat().st_size,
                          sha256=digest.hexdigest()),indent=2))


if __name__=='__main__':
    main()
