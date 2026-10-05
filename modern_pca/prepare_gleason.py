"""Join released labels, exclude NC, audit splits and cache legacy StainTools.

No model construction, pseudo-labels, purity filters, or subpatch relabelling.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import importlib.metadata
import json
from pathlib import Path
import zipfile

from .gleason import CLASSES, REFERENCE, REFERENCE_SHA, sha, write_json, write_csv, provenance

ARCHIVES = {
    'Annotations.zip': '7dc40c923f77a251ae564977647938b1',
    'Patches.zip': 'dd7c63e595793aad46c07b0f0578e0b1',
    'NormalizedPatches.zip': '74ebb5ceed5a8d8a0b0e2df960f7bd94',
    'NormalizedSICAPv2.zip': '61be06796ede2d8c48f01b45b7306a16',
    'NormalizedSICAPv2_Annotations.zip': '0cbd85834d5ba72cdceb2d76066c1914',
}


def extract(archive, destination):
    import hashlib
    if archive.name in ARCHIVES:
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'md5').hexdigest()
        if digest != ARCHIVES[archive.name]:
            raise ValueError(f'Archive checksum mismatch: {archive}')
    archive_sha = sha(archive)
    marker = destination / '.archive.json'
    if marker.exists():
        if json.loads(marker.read_text())['sha256'] != archive_sha:
            raise ValueError('Extraction root belongs to a different archive')
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe ZIP member: {info.filename}')
        z.extractall(destination)
    write_json(marker, {'sha256': archive_sha})


def one(root, name):
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} under {root}; found {len(matches)}')
    return matches[0]


def crowd_labels(root):
    for split, filename in [('train','train.csv'), ('validation','val.csv'), ('test','test.csv')]:
        with one(root, filename).open(newline='', encoding='utf-8-sig') as stream:
            for record in csv.DictReader(stream):
                name = record['Patch filename'].strip()
                key = 'ground truth' if split == 'test' else 'MV'
                label = int(record[key])
                if label not in range(4):
                    raise ValueError(f'Invalid CrowdGleason label: {label}')
                votes = [int(record[f'marker{i}']) for i in range(1,8)]
                if any(v not in (-1,0,1,2,3) for v in votes):
                    raise ValueError('Invalid annotator label')
                yield dict(dataset='crowd', split=split, name=Path(name).stem,
                           source_label=label, label_source=key,
                           source_wsi=name.split('_row_')[0], votes=json.dumps(votes), source_g4c='')


def sicap_labels(root):
    from openpyxl import load_workbook
    # Use released Val1 partition when given the original SICAP layout.
    original = list(root.rglob('partition'))
    if original:
        if len(original) != 1:
            raise ValueError('Ambiguous SICAP partition root')
        p = original[0]
        paths = [('train', p/'Validation/Val1/Train.xlsx'),
                 ('validation', p/'Validation/Val1/Test.xlsx'), ('test', p/'Test/Test.xlsx')]
    else:
        paths = [(s, one(root, f)) for s,f in [('train','Train.xlsx'), ('validation','Val.xlsx'), ('test','Test.xlsx')]]
    for split, path in paths:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            iterator = workbook.active.iter_rows(values_only=True)
            header = next(iterator)
            for cells in iterator:
                row = dict(zip(header,cells))
                if not row.get('image_name'):
                    continue
                classes = [int(row.get(key) or 0) for key in ('NC','G3','G4','G5')]
                if any(x not in (0,1) for x in classes) or sum(classes) != 1:
                    raise ValueError(f'Invalid SICAP one-hot class: {row}')
                name = Path(str(row['image_name'])).stem
                yield dict(dataset='sicap', split=split, name=name,
                           source_label=classes.index(1), label_source='published_patch_class',
                           source_wsi=name.split('_Block_')[0], votes='', source_g4c=row.get('G4C') or 0)
        finally:
            workbook.close()


def image_index(root):
    index = {}
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff') or not path.is_file():
            continue
        if 'masks' in [s.lower() for s in path.relative_to(root).parts]:
            continue
        if path.stem in index:
            raise ValueError(f'Duplicate image identity: {path.stem}')
        index[path.stem] = path
    return index


def audit_splits(rows):
    seen = {}
    hashes = {}
    slides = {}
    for r in rows:
        key = (r['dataset'], r['name'])
        if key in seen:
            raise ValueError(f'Duplicate sample across annotations: {key}')
        seen[key] = r['split']
        hashes.setdefault(r['file_sha256'], set()).add(r['split'])
        slides.setdefault((r['dataset'],r['source_wsi']), set()).add(r['split'])
    if any(len(s) > 1 for s in hashes.values()):
        raise ValueError('Identical image content crosses split boundaries')
    overlaps = [{'dataset': k[0], 'source_wsi': k[1], 'splits':sorted(v)}
                for k,v in slides.items() if len(v)>1]
    return {'slide_overlaps':overlaps,
            'duplicate_content_rows': len(rows)-len(hashes),
            'patient_independence': 'not established from filename prefixes; source partitions retained'}


def tumour_rows(rows):
    return [dict(r, label=r['source_label']-1, class_name=CLASSES[r['source_label']-1])
            for r in rows if r['source_label'] != 0]


def init_worker(reference, cache):
    global _normalizers, _cache
    from .evaluate_paper import create_legacy_normalizer
    _normalizers = create_legacy_normalizer(Path(reference))
    _cache = Path(cache)


def prepare_image(row):
    import numpy as np
    from PIL import Image
    from .evaluate_paper import preprocess_legacy
    target = _cache / (row['sample_id'].replace(':','_') + '.png')
    pixels = preprocess_legacy(row, Path(row['path']), _normalizers)
    # Normalizer emits uint8; cache losslessly before training-time /255 scaling.
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(target)
    return dict(row, cache_file=target.name, cache_sha256=sha(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', type=Path)
    p.add_argument('--extracted', type=Path)
    p.add_argument('--crowd-images', type=Path)
    p.add_argument('--crowd-annotations', type=Path)
    p.add_argument('--sicap-images', type=Path)
    p.add_argument('--sicap-annotations', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['raw-crowd','normalized'], default='raw-crowd')
    p.add_argument('--allow-prenormalized', action='store_true', help='Explicitly accept source normalization before the legacy pipeline')
    p.add_argument('--crowd-label', choices=['MV'], default='MV')
    p.add_argument('--tumour-only', action='store_true', help='Always enforced')
    p.add_argument('--preprocessing', choices=['legacy-staintools'], default='legacy-staintools')
    p.add_argument('--image-size', type=int, choices=[350], default=350)
    p.add_argument('--stain-reference', type=Path, default=REFERENCE)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-only', action='store_true', help='Allow missing images for structural sample checks; barred from production')
    p.add_argument('--audit-only', action='store_true', help='Joins and image audit only; no StainTools cache')
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.archives:
        if not args.extracted or not args.allow_prenormalized:
            p.error('Archive route uses normalized SICAP; requires --extracted and --allow-prenormalized')
        names = {'crowd-images': 'Patches.zip' if args.variant=='raw-crowd' else 'NormalizedPatches.zip',
                 'crowd-annotations':'Annotations.zip', 'sicap-images':'NormalizedSICAPv2.zip',
                 'sicap-annotations':'NormalizedSICAPv2_Annotations.zip'}
        for key, filename in names.items():
            destination = args.extracted / key
            extract(args.archives / filename, destination)
            setattr(args, key.replace('-','_'), destination)
    sources = [args.crowd_images,args.crowd_annotations,args.sicap_images,args.sicap_annotations]
    if not all(sources):
        p.error('Provide --archives/--extracted or all four explicit image/annotation roots')
    for source in sources:
        a,b = args.output.resolve(),source.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            p.error('Output must be separate from source directories')
    args.output.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    rows, missing, counts = [], [], Counter()
    for labels, root in [(crowd_labels(args.crowd_annotations),args.crowd_images),
                         (sicap_labels(args.sicap_annotations),args.sicap_images)]:
        index = image_index(root)
        for item in labels:
            counts[f"{item['dataset']}/{item['split']}/source_{item['source_label']}"] += 1
            if item['name'] not in index:
                missing.append(item['name'])
                continue
            path = index[item['name']]
            with Image.open(path) as image:
                image.load()
                width,height = image.size
                mode = image.mode
            rows.append(dict(item, sample_id=item['dataset']+':'+item['name'],
                             path=str(path.resolve()), file_sha256=sha(path),
                             source_width=width, source_height=height, source_mode=mode))
    audit = dict(provenance(), source_counts=dict(counts), missing_images=len(missing),
                 missing_examples=missing[:20], split_audit=audit_splits(rows),
                 sample_only=args.sample_only, prenormalized_sources_accepted=args.allow_prenormalized)
    write_json(args.output/'audit.json', audit)
    if missing and not args.sample_only:
        raise ValueError('Missing images: see audit.json; production requires complete joins')
    rows = tumour_rows(rows)
    audit['retained_counts'] = dict(Counter(f"{r['dataset']}/{r['split']}/{r['class_name']}" for r in rows))
    write_json(args.output/'audit.json',audit)
    if args.audit_only:
        write_json(args.output/'audit_rows.json', rows)
        return
    if sha(args.stain_reference) != REFERENCE_SHA:
        raise ValueError('Stain reference hash mismatch')
    cache = args.output/'cache'
    cache.mkdir()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.stain_reference),str(cache))) as pool:
        prepared = []
        for item in pool.map(prepare_image, rows, chunksize=16):
            prepared.append(item)
            if len(prepared) % 100 == 0 or len(prepared) == len(rows):
                print(f'Normalized {len(prepared)}/{len(rows)} tumour patches', flush=True)
    manifest_dir = args.output/'manifests'
    manifest_dir.mkdir()
    fields = list(prepared[0]) if prepared else []
    if not prepared:
        raise ValueError('No tumour images')
    manifests = {}
    for name in ('crowd','sicap'):
        for split in ('train','validation','test'):
            path = manifest_dir/f'{name}_{split}.csv'
            selected = [r for r in prepared if r['dataset']==name and r['split']==split]
            if not selected and not args.sample_only:
                raise ValueError(f'Empty partition {name}/{split}')
            write_csv(path,selected,fields)
            manifests[path.name] = sha(path)
    write_json(args.output/'prepared.json',dict(audit, manifests=manifests,
               class_order=CLASSES, reference_sha256=REFERENCE_SHA,
               preprocessing='RGB -> LANCZOS 350 -> legacy brightness -> StainTools Macenko -> uint8 cache -> float32 /255',
               staintools=importlib.metadata.version('staintools'), complete=True))


if __name__ == '__main__':
    main()
