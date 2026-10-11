"""Isolated SICAPv2 ingestion contracts. No dataset access on import."""
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath

from tools.inspect_sicapv2 import read_xlsx

ARCHIVE_SHA='a28aa6a0e3831217bf31575cecfc1846607ace7ff25a4392f07052bbc7355916'
ARCHIVE_BYTES=2159052394
DATASET_URL='https://data.mendeley.com/datasets/9xxm58dvs3/2'
DOWNLOAD_URL='https://data.mendeley.com/public-files/datasets/9xxm58dvs3/files/6ab087a7-ca89-47ac-9698-f6546bb50f98/file_downloaded'


def table(payload,header):
    sheets=read_xlsx(payload)
    if len(sheets)!=1: raise ValueError('Expected one worksheet')
    rows=next(iter(sheets.values()))
    if not rows or rows[0]!=header: raise ValueError('Unexpected metadata schema')
    if any(len(row)!=len(header) for row in rows[1:]): raise ValueError('Incomplete metadata row')
    return [dict(zip(header,row)) for row in rows[1:]]


def primary_label(row):
    values=[row[key] for key in ('NC','G3','G4','G5')]
    if any(type(value) is not int or value not in (0,1) for value in values) or sum(values)!=1:
        raise ValueError('Ambiguous/non-one-hot official primary label')
    return ('NC','G3','G4','G5')[values.index(1)],int(values[0]==0)


def safe_member(name):
    path=PurePosixPath(name)
    if '\\' in name or path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='SICAPv2':
        raise ValueError('Unsafe archive member')
    return path


def sample_order(rows):
    return sorted(rows,key=lambda row:(hashlib.sha256(('SICAPv2-epoch15-v1|42|'+row['sample_id']).encode()).hexdigest(),row['sample_id']))


def patient_partition(train,test):
    overlap={row['patient_id'] for row in train}&{row['patient_id'] for row in test}
    if overlap: raise ValueError('Patient overlap between official partitions')
    return dict(train_patients=len({r['patient_id'] for r in train}),test_patients=len({r['patient_id'] for r in test}),overlap=0)


def mask_summary(pixels):
    import numpy as np
    # Version 2 bytes disagree with the bundled Version 1 readme. Inventory
    # literal values; never infer training labels or biological mixtures here.
    if pixels.ndim!=2 or pixels.dtype!=np.uint8: raise ValueError('Expected uint8 grayscale mask')
    counts=np.bincount(pixels.ravel(),minlength=256)
    return dict(**{f'mask_value_{i}':int(counts[i]) for i in range(6)},
                mask_other=int(sum(counts[6:])),
                mixed_zero_nonzero=bool(counts[0] and sum(counts[1:])),
                multiple_nonzero_values=bool(np.count_nonzero(counts[1:])>1),
                majority_nonzero_value=int(np.argmax(counts[1:])+1) if sum(counts[1:]) else 0)


def duplicate_groups(rows,key='file_sha256'):
    groups={}
    for row in rows: groups.setdefault(row[key],[]).append(row['sample_id'])
    return [values for values in groups.values() if len(values)>1]
