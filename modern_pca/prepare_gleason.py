"""Prepare published Crowd/SICAP labels or an unsplit Gleason2019 tile corpus.

No model construction, pseudo-labels, purity thresholds, or invented case grades.
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
import io
import hashlib
import re
import urllib.request

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


def slide_grades(path):
    """Read SICAP slide-level grades; these are not per-patch target pairs."""
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    result = {}
    try:
        rows = workbook.active.iter_rows(values_only=True)
        header = next(rows)
        required = {'slide_id', 'patient_id', 'Gleason_primary', 'Gleason_secondary'}
        if not required.issubset(header):
            raise ValueError('Missing SICAP slide-grade columns')
        for cells in rows:
            row = dict(zip(header, cells))
            if row.get('slide_id') is None:
                continue
            name = str(row['slide_id']).strip()
            pair = (row['Gleason_primary'], row['Gleason_secondary'])
            if any(v not in (0, 3, 4, 5) for v in pair):
                raise ValueError(f'Invalid SICAP slide grade for {name}: {pair}')
            if name in result:
                raise ValueError(f'Duplicate SICAP slide grade: {name}')
            result[name] = dict(region_primary=int(pair[0]), region_secondary=int(pair[1]),
                                patient_id='' if row['patient_id'] is None else str(row['patient_id']),
                                region_grade_source='SICAP wsi_labels')
    finally:
        workbook.close()
    return result


def assign_region_role(row, grades):
    metadata = grades.get(row['source_wsi'], {}) if row['dataset'] == 'sicap' else {}
    result = dict(row, region_primary='', region_secondary='', patient_id='',
                  region_grade_source='', region_role='unknown')
    result.update(metadata)
    if metadata:
        primary, secondary = metadata['region_primary'], metadata['region_secondary']
        if primary in (3, 4, 5) and secondary in (3, 4, 5):
            result['region_role'] = ('mixed' if primary != secondary else
                                     'pure' if row['source_label'] == primary - 2 else 'conflict')
        elif row['source_label'] != 0:
            result['region_role'] = 'conflict'
    return result


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


GLEASON_MASK_URL = ('https://data.mendeley.com/public-files/datasets/s384w7kv78/files/'
                   '4d3b905f-8b97-4b5d-bd24-915823501f7f/file_downloaded')
GLEASON_MASK_SHA = 'feba9d1620d4dec1926ae142ced7afaa91d028630a6fb160b0c110e36b95625a'


def download_gleason2019(root):
    """Official public training JPEGs plus checksum-pinned public mask mirror.

    Sync's public-link transport is encrypted; no account credentials are used.
    Fail closed if its public client protocol changes. Never log signed URLs/keys.
    """
    import base64
    import time
    import urllib.parse
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    root.mkdir(parents=True, exist_ok=True)
    ua = 'Mozilla/5.0 Gleason2019-dataset-inspection'
    def request(url, body=None):
        return urllib.request.urlopen(urllib.request.Request(url,
            data=None if body is None else json.dumps(body).encode(),
            headers={'User-Agent': ua, 'Content-Type': 'application/json'}), timeout=90)
    def post(command, body):
        with request('https://ln.sync.com/api/v1/' + command, body) as response:
            value = json.load(response)
        if value.get('success') != 1:
            raise RuntimeError(f'Public Sync API failed: {command}')
        return value
    def decrypt(value, key):
        raw = base64.b64decode(value.split(':', 1)[-1])
        cipher = Cipher(algorithms.AES(key), modes.GCM(raw[:12], raw[-12:], min_tag_length=12)).decryptor()
        return cipher.update(raw[12:-12]) + cipher.finalize()
    archive = root / 'masks.zip'
    if not archive.exists() or sha(archive) != GLEASON_MASK_SHA:
        partial = archive.with_suffix('.part')
        with request(GLEASON_MASK_URL) as response, partial.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if sha(partial) != GLEASON_MASK_SHA:
            raise ValueError('Gleason2019 mask archive checksum mismatch')
        partial.replace(archive)
    link_id, public_key_text = '2312d2d50', 'gtv9skii-qsj4j74i-72vhbxm8-2fzjztz7'
    with request('https://ln.sync.com') as response:
        html = response.read().decode()
    match = re.search(r'src="([^"]*main\.[^"]+\.js)', html)
    if not match:
        raise RuntimeError('Sync public frontend changed: cannot locate client')
    with request(urllib.parse.urljoin('https://ln.sync.com/', match.group(1))) as response:
        client = response.read().decode()
    begin = client.index('compatDatakeyEncrypt(a){')
    end = client.index(']:[', begin)
    pem = '\n'.join(s for s in re.findall(r'"([^"\n]+)"', client[begin:end])
                    if not s.startswith('compat')) + '\n'
    rsa_key = serialization.load_pem_public_key(pem.encode())
    def rsa(value):
        return base64.b64encode(rsa_key.encrypt(value.encode(), padding.PKCS1v15())).decode().rstrip('=')
    metadata = post('linkpathlist', dict(publink_id=link_id, sync_id=0, passwordlock=''))
    key = hashlib.pbkdf2_hmac('sha256', public_key_text.encode(), bytes.fromhex(metadata['salt']),
                            metadata['iterations'], 64)
    items = metadata['pathitems']
    names = set()
    for item in items:
        item['filename'] = decrypt(item['enc_share_name'], key[32:]).decode()
        if not re.fullmatch(r'slide\d+_core\d+\.jpg', item['filename']) or item['filename'] in names:
            raise ValueError('Unexpected or duplicate public training image name')
        names.add(item['filename'])
    public = [dict({k: r[k] for k in ('share_id', 'blob_id', 'sync_id', 'size')},
                   ext=base64.b64encode(b'jpg').decode(), link_cachekey=link_id, user_id=0) for r in items]
    data = post('pathdata', dict(pathitems=public))
    images = root / 'images'
    images.mkdir(exist_ok=True)
    ledger_path = root / 'downloads.json'
    old = json.loads(ledger_path.read_text()).get('images', {}) if ledger_path.exists() else {}
    ledger = {}
    from PIL import Image
    for count, item in enumerate(items, 1):
        name = item['filename']
        target = images / name
        if not (target.exists() and target.stat().st_size == item['size'] and
                old.get(name, {}).get('sha256') == sha(target)):
            file_key = decrypt(data['datakeys'][str(item['sync_id'])]['enc_data_key'], key[:32])
            params = dict(sharelink_id=item['sync_id'], linkoid=metadata['oid'], linkcachekey=link_id,
                mode=101, datakey=rsa(base64.b64encode(file_key).decode()),
                header1=base64.b64encode(b'Content-Type: image/jpeg').decode().rstrip('='),
                header2=base64.b64encode(f'Content-Disposition: attachment; filename="{name}";'.encode()).decode().rstrip('='),
                uagent=hashlib.sha1(ua.encode()).hexdigest(), ipaddress='s',
                errurl=rsa(base64.b64encode(f'https://ln.sync.com/dl/{link_id}/{public_key_text}'.encode()).decode()),
                timestamp=int(time.time() * 1000), engine='ln-3.1.38')
            signed = post('linksignrequest', dict(req=params))
            params = signed['response']
            params['cachekey'] = item['cachekey']
            if signed.get('pltoken'):
                params['pltoken'] = signed['pltoken']
            host = metadata['servers_compat'][0]
            if not re.fullmatch(r'[A-Za-z0-9.-]+\.syncusercontent[0-9]*\.com', host):
                raise ValueError('Unexpected Sync download host')
            # The public client signs literal query values, not urlencode output.
            url = f'https://{host}/p/{name}?' + '&'.join(f'{k}={v}' for k, v in params.items())
            partial = target.with_suffix('.part')
            with request(url) as response, partial.open('wb') as output:
                if 'image' not in response.headers.get('Content-Type', ''):
                    raise ValueError('Public download did not return an image')
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if partial.stat().st_size != item['size']:
                raise ValueError(f'Download byte count mismatch: {name}')
            with Image.open(partial) as image:
                if image.format != 'JPEG':
                    raise ValueError(f'Not a JPEG: {name}')
                image.verify()
            partial.replace(target)
        ledger[name] = dict(bytes=item['size'], sha256=sha(target))
        write_json(ledger_path, dict(images={**old, **ledger}, mask_sha256=GLEASON_MASK_SHA,
                                    client_sha256=hashlib.sha256(client.encode()).hexdigest()))
        print(f'Downloaded/verified {count}/{len(items)} cores', flush=True)
    return images, archive


def consensus_mask(masks):
    """Vote among available masks, including background; ties unresolved.

    Raw value 6 abstains at that pixel; it never invalidates other experts.
    This is an explicit preparation policy, not an asserted challenge ground truth.
    """
    import numpy as np
    values = np.stack(masks)
    if not np.isin(values, [0, 1, 3, 4, 5, 6]).all():
        raise ValueError('Unexpected mask encoding')
    counts = np.stack([(values == label).sum(axis=0) for label in (0, 1, 3, 4, 5)])
    maxima = counts.max(axis=0)
    valid = (maxima > 0) & ((counts == maxima).sum(axis=0) == 1)
    output = np.zeros(values.shape[1:], dtype=np.uint8)
    output[valid] = np.array([0, 1, 3, 4, 5], dtype=np.uint8)[counts.argmax(axis=0)[valid]]
    unresolved = (values != 0).any(axis=0) & ~valid
    return output, unresolved


def grouped_core_split(cores, seed=42):
    """Freeze an approximately 80/20 split by filename slide, never by patch."""
    import random
    groups = sorted({core.split('_core')[0] for core in cores})
    if len(groups) < 2:
        raise ValueError('Need at least two slide groups for train/validation')
    random.Random(seed).shuffle(groups)
    heldout = set(groups[:max(1, round(len(groups) * .2))])
    return {core: 'validation' if core.split('_core')[0] in heldout else 'train'
            for core in sorted(cores)}


def prepare_gleason2019(args):
    """Prepare independent Gleason2019 manifests with a single-pattern-core proxy."""
    import numpy as np
    from PIL import Image
    from .evaluate_paper import create_legacy_normalizer
    from .gleason_regions import preprocess_image
    root = args.gleason2019_root.resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError('Prepared output already exists; select a new output directory')
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError('Gleason2019 raw and prepared roots must be separate')
    images, archive = download_gleason2019(root)
    if sha(archive) != GLEASON_MASK_SHA:
        raise ValueError('Mask checksum mismatch')
    mask_root = root / 'masks'
    with zipfile.ZipFile(archive) as outer:
        archives = [n for n in outer.namelist() if n.endswith('.zip')]
        if len(archives) != 6:
            raise ValueError('Expected six expert mask archives')
        for member in archives:
            expert = int(re.search(r'Maps(\d)', member).group(1))
            target = mask_root / f'expert{expert}'
            nested = root / f'expert{expert}.zip'
            with nested.open('wb') as output:
                output.write(outer.read(member))
            extract(nested, target)
    index = {}
    for path in mask_root.rglob('*_classimg_nonconvex.png'):
        core = path.name.removesuffix('_classimg_nonconvex.png')
        index.setdefault(core, []).append(path)
    destination.mkdir(parents=True, exist_ok=False)
    cache = destination / 'cache'
    cache.mkdir()
    if sha(args.stain_reference) != REFERENCE_SHA:
        raise ValueError('Stain reference checksum mismatch')
    normalizers = None if args.audit_only else create_legacy_normalizer(args.stain_reference)
    rows, counts, failures, core_records = [], Counter(), [], []
    number = 0
    paths = sorted(images.glob('*.jpg'))
    if not paths:
        raise ValueError('No Gleason2019 training images')
    orphan_masks = sorted(set(index) - {p.stem for p in paths})
    split = grouped_core_split([p.stem for p in paths], getattr(args, 'seed', 42))
    write_json(destination / 'split.json', dict(seed=getattr(args, 'seed', 42),
               grouping='filename slide identifier; patient independence unverified', cores=split))
    try:
        for number, path in enumerate(paths, 1):
            core = path.stem
            masks = sorted(index.get(core, []))
            if not masks:
                counts['cores_without_masks'] += 1
                core_records.append(dict(core=core, split=split[core], experts=0, patterns=[],
                                         region_role='unknown', excluded_reason='no masks'))
                continue
            with Image.open(path) as original:
                original.load()
                arrays = []
                for mask_path in masks:
                    with Image.open(mask_path) as mask:
                        if mask.mode != 'L' or mask.size != original.size:
                            raise ValueError(f'Mask geometry/encoding mismatch: {mask_path.name}')
                        arrays.append(np.asarray(mask).copy())
                image_sha = sha(path)
                mask_shas = json.dumps({str(p.relative_to(mask_root)): sha(p) for p in masks}, sort_keys=True)
                width, height = original.size
                counts['discarded_edge_pixels'] += width * height - (width // 600 * 600) * (height // 600 * 600)
                # Core eligibility includes the edge regions, not just cached tiles.
                core_patterns, core_unresolved, invalid_votes = set(), 0, 0
                for y in range(0, height, 600):
                    for x in range(0, width, 600):
                        tiles = [m[y:y+600, x:x+600] for m in arrays]
                        voted, unresolved = consensus_mask(tiles)
                        core_patterns.update(int(gp) for gp in (3, 4, 5) if (voted == gp).any())
                        core_unresolved += int(unresolved.sum())
                        invalid_votes += sum(int((m == 6).sum()) for m in tiles)
                role = 'pure' if len(core_patterns) == 1 else 'mixed' if core_patterns else 'unknown'
                grade = next(iter(core_patterns)) if role == 'pure' else ''
                core_records.append(dict(core=core, split=split[core], experts=len(masks),
                    patterns=sorted(core_patterns), region_role=role, unresolved_pixels=core_unresolved,
                    invalid_value6_votes=invalid_votes, image_sha256=image_sha, mask_sha256=mask_shas))
                for y in range(0, height - 599, 600):
                    for x in range(0, width - 599, 600):
                        voted, unresolved = consensus_mask([m[y:y+600, x:x+600] for m in arrays])
                        hist = np.bincount(voted.ravel(), minlength=7)
                        patterns = [gp for gp in (3, 4, 5) if hist[gp]]
                        status = ('mixed' if len(patterns) > 1 else 'single_gp' if patterns else
                                  'unresolved' if unresolved.any() else 'non_tumour')
                        counts[status] += 1
                        name = f'{core}_x{x}_y{y}'
                        row = dict(sample_id='gleason2019:' + name, dataset='gleason2019', source_core=core,
                            source_slide=core.split('_core')[0], source_wsi=core.split('_core')[0],
                            split=split[core], region_role=role, region_primary=grade, region_secondary=grade,
                            region_grade_source='consensus_single_pattern_core_proxy', patient_id='',
                            x=x, y=y, native_size=600, output_size=350, experts=len(masks),
                            status=status, class_name=f'GP{patterns[0]}' if status == 'single_gp' else '',
                            label=patterns[0]-3 if status == 'single_gp' else '',
                            label_source='available_expert_pixel_plurality',
                            gp3_pixels=int(hist[3]), gp4_pixels=int(hist[4]), gp5_pixels=int(hist[5]),
                            benign_pixels=int(hist[1]), background_pixels=int(hist[0]),
                            unresolved_pixels=int(unresolved.sum()), unresolved_fraction=float(unresolved.mean()),
                            invalid_value6_votes=sum(int((m[y:y+600, x:x+600] == 6).sum()) for m in arrays),
                            image_sha256=image_sha, file_sha256=image_sha,
                            mask_sha256=mask_shas, cache_file='', cache_sha256='')
                        # Partial uncertainty does not veto resolved tumour evidence.
                        # Multiple GPs are cached without a supervised target.
                        if status in ('single_gp', 'mixed') and not args.audit_only:
                            tile = original.crop((x, y, x+600, y+600))
                            pixels = preprocess_image(tile, normalizers)
                            target = cache / (name + '.png')
                            Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(target)
                            row.update(cache_file=target.name, cache_sha256=sha(target))
                        rows.append(row)
            print(f'Prepared {number}/{len(paths)} cores', flush=True)
    except Exception as error:
        failures.append(dict(error=type(error).__name__, message=str(error)))
        raise
    finally:
        if rows:
            write_csv(destination / 'patch_inventory.csv', rows)
        write_json(destination / 'gleason2019.json', dict(provenance(), dataset='Gleason2019',
            complete=not failures and number == len(paths), failures=failures, counts=dict(counts),
            audit_only=args.audit_only, core_count=len(paths), orphan_mask_cores=orphan_masks,
            mask_archive_sha256=GLEASON_MASK_SHA, reference_sha256=REFERENCE_SHA,
            geometry='native 600x600, stride 600, discard incomplete edges; image LANCZOS 350x350',
            voting='all available expert labels including background; unique plurality; ties and raw 6 unresolved',
            training_ready=False, reason='See prepared.json for eligibility and class coverage',
            preprocessing='existing legacy brightness + StainTools Macenko; no stromal decontamination'))
    write_json(destination / 'cores.json', core_records)
    metadata = json.loads((destination / 'gleason2019.json').read_text())
    metadata['voting'] = 'unique plurality including background; ties unresolved; value 6 abstains'
    eligible = [r for r in rows if r['status'] in ('single_gp', 'mixed')]
    manifests = destination / 'manifests'
    manifests.mkdir()
    hashes = {}
    fields = list(rows[0]) if rows else ['dataset', 'split', 'label', 'class_name']
    for partition in ('train', 'validation'):
        selected = [r for r in eligible if r['split'] == partition and
                    (partition == 'train' or r['status'] == 'single_gp')]
        path = manifests / f'gleason2019_{partition}.csv'
        write_csv(path, selected, fields)
        hashes[path.name] = sha(path)
    from .gleason import select_training, assert_training_isolated
    seed_rows = select_training([r for r in eligible if r['split'] == 'train'], 'pure')
    validation = [r for r in eligible if r['split'] == 'validation' and r['status'] == 'single_gp']
    assert_training_isolated([r for r in eligible if r['split'] == 'train'], validation)
    ready = not args.audit_only and {r['label'] for r in seed_rows} == {0, 1, 2} and bool(validation)
    metadata.update(training_ready=ready, reason='ready' if ready else 'Need all three seed GP classes and validation; audit-only cannot train',
        seed_class_counts=dict(Counter(r['class_name'] for r in seed_rows)),
        validation_class_counts=dict(Counter(r['class_name'] for r in validation)),
        datasets=['gleason2019'], manifests=hashes, class_order=CLASSES, sample_only=args.audit_only,
        region_grade_sha256=sha(destination / 'cores.json'), split_sha256=sha(destination / 'split.json'),
        prenormalized_sources_accepted=False, reference_sha256=REFERENCE_SHA,
        evaluation_split='validation', seed_definition='single-pattern-core consensus proxy, not verified pure patient cases')
    write_json(destination / 'gleason2019.json', metadata)
    write_json(destination / 'prepared.json', metadata)
    print(f'Gleason2019 training-ready: {ready}; seed classes: {metadata["seed_class_counts"]}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', type=Path)
    p.add_argument('--gleason2019-root', type=Path,
                   help='Automatically download and prepare official Gleason2019 training cores here')
    p.add_argument('--extracted', type=Path)
    p.add_argument('--crowd-images', type=Path)
    p.add_argument('--crowd-annotations', type=Path)
    p.add_argument('--sicap-images', type=Path)
    p.add_argument('--sicap-annotations', type=Path)
    p.add_argument('--sicap-wsi-labels', type=Path,
                   help='SICAP wsi_labels.xlsx; required for documented pure/mixed cohorts')
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
    if args.gleason2019_root:
        prepare_gleason2019(args)
        return
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
    grade_path = args.sicap_wsi_labels
    if grade_path is None:
        found = list(args.sicap_annotations.rglob('wsi_labels.xlsx'))
        if len(found) > 1:
            p.error('Ambiguous wsi_labels.xlsx; provide --sicap-wsi-labels')
        grade_path = found[0] if found else None
    grades = slide_grades(grade_path) if grade_path else {}
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
            item = assign_region_role(item, grades)
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
                 region_grade_sha256=sha(grade_path) if grade_path else None,
                 missing_examples=missing[:20], split_audit=audit_splits(rows),
                 sample_only=args.sample_only, prenormalized_sources_accepted=args.allow_prenormalized)
    write_json(args.output/'audit.json', audit)
    if missing and not args.sample_only:
        raise ValueError('Missing images: see audit.json; production requires complete joins')
    rows = tumour_rows(rows)
    audit['region_role_counts'] = dict(Counter(f"{r['dataset']}/{r['split']}/{r['region_role']}" for r in rows))
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
