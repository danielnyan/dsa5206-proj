"""Extract and audit authenticated SICAPv2 only. Reads VAL manifests, never their images."""
import csv
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image
from modern_pca import sicapv2 as s, reimplementation as r
from tools.inspect_sicapv2 import ROOT


def guard():
    def audit(event,args):
        if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if 'val_dataset_2' in os.fsdecode(args[0]).lower(): raise RuntimeError('VAL2 image access forbidden')
    sys.addaudithook(audit)


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--audit-extracted',action='store_true')
    args=parser.parse_args()
    guard(); started=time.perf_counter()
    archive_path=ROOT/'SICAPv2.zip'
    assert archive_path.stat().st_size==s.ARCHIVE_BYTES
    r.sha256_file(archive_path,s.ARCHIVE_SHA)
    destination=ROOT/'extracted'
    audit_dir=ROOT/'audit'
    if not args.audit_extracted:
        destination.mkdir(exist_ok=False); audit_dir.mkdir(exist_ok=False)
    metadata={}; entries=[]
    if args.audit_extracted:
        with (audit_dir/'archive_members.csv').open() as stream:
            entries=[dict(row,bytes=int(row['bytes']),compressed_bytes=int(row['compressed_bytes'])) for row in csv.DictReader(stream)]
    with zipfile.ZipFile(archive_path) as archive:
        infos=archive.infolist()
        assert len({info.filename for info in infos})==len(infos)
        for info in infos:
            member=s.safe_member(info.filename)
            if args.audit_extracted:
                if Path(info.filename).suffix in ('.xlsx','.txt'): metadata[info.filename]=archive.read(info)
                continue
            if (info.external_attr>>16)&0o170000==0o120000: raise ValueError('Archive symlink forbidden')
            target=destination/str(member)
            assert target.resolve().is_relative_to(destination.resolve())
            if info.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
            payload=archive.read(info)  # Includes ZIP CRC verification.
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream: stream.write(payload)
            digest=hashlib.sha256(payload).hexdigest()
            entries.append(dict(path=str(member),bytes=len(payload),sha256=digest,compressed_bytes=info.compress_size))
            if target.suffix in ('.xlsx','.txt'): metadata[str(member)]=payload
            if len(entries)%5000==0: print('extracted',len(entries),flush=True)
    if not args.audit_extracted: write_csv(audit_dir/'archive_members.csv',entries)
    expected_hashes={entry['path']:entry['sha256'] for entry in entries}
    slides=s.table(metadata['SICAPv2/wsi_labels.xlsx'],['slide_id','patient_id','Gleason_primary','Gleason_secondary'])
    slide_map={str(row['slide_id']):row for row in slides}; assert len(slide_map)==len(slides)
    tables={}; official={}; conflicts=[]
    for name,payload in metadata.items():
        if name.endswith(('/Train.xlsx','/Test.xlsx')):
            rows=s.table(payload,['image_name','NC','G3','G4','G5','G4C'])
            assert len({row['image_name'] for row in rows})==len(rows)
            tables[name]=rows
            for row in rows:
                label,binary=s.primary_label(row)
                if row['image_name'] in official and official[row['image_name']]!=(label,binary): conflicts.append(row['image_name'])
                official[row['image_name']]=(label,binary)
    assert not conflicts
    train_names={row['image_name'] for row in tables['SICAPv2/partition/Test/Train.xlsx']}
    test_names={row['image_name'] for row in tables['SICAPv2/partition/Test/Test.xlsx']}
    assert not train_names&test_names
    records=[]; errors=[]; image_info=[]; mask_info=[]
    source=destination/'SICAPv2'
    image_paths=sorted((source/'images').glob('*.jpg'))
    assert {p.stem for p in image_paths}=={p.stem for p in (source/'masks').glob('*.png')}
    for index,path in enumerate(image_paths):
        try:
            payload=path.read_bytes()
            assert hashlib.sha256(payload).hexdigest()==expected_hashes[str(path.relative_to(destination))]
            with Image.open(io.BytesIO(payload)) as image:
                image.load(); pixels=np.asarray(image)
                info=dict(width=image.width,height=image.height,mode=image.mode,format=image.format,
                          info=str(image.info),exif=str(dict(image.getexif())))
                assert pixels.ndim==3 and pixels.shape[2]==3
            mask_path=source/'masks'/f'{path.stem}.png'
            r.sha256_file(mask_path,expected_hashes[str(mask_path.relative_to(destination))])
            with Image.open(mask_path) as mask:
                mask.load(); mp=np.asarray(mask)
                mi=dict(width=mask.width,height=mask.height,mode=mask.mode,format=mask.format)
            assert mp.shape==pixels.shape[:2]
            ms=s.mask_summary(mp)
            slide_id=path.name.split('_Block_Region_')[0]
            slide=slide_map[slide_id]
            label,binary=official.get(path.name,('unlisted',-1))
            records.append(dict(sample_id='SICAPv2:'+path.name,filename=path.name,slide_id=slide_id,
                patient_id=str(slide['patient_id']),primary_grade=slide['Gleason_primary'],secondary_grade=slide['Gleason_secondary'],
                official_label=label,ground_truth_code=binary,partition='train' if path.name in train_names else 'test' if path.name in test_names else 'unlisted',
                file_sha256=hashlib.sha256(payload).hexdigest(),pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                mask_sha256=r.sha256_file(mask_path),width=info['width'],height=info['height'],mode=info['mode'],
                mean_r=float(pixels[:,:,0].mean()),mean_g=float(pixels[:,:,1].mean()),mean_b=float(pixels[:,:,2].mean()),
                bright_pixel_fraction=float(np.mean(np.all(pixels>=240,axis=2))),std_rgb=float(pixels.std()),**ms))
            image_info.append(info); mask_info.append(mi)
        except Exception as error: errors.append(dict(filename=path.name,error=str(error)))
        if (index+1)%2000==0: print('decoded',index+1,'errors',len(errors),flush=True)
    assert set(official).issubset({p.name for p in image_paths})
    write_csv(audit_dir/'images.csv',records)
    train=[row for row in records if row['partition']=='train']; test=[row for row in records if row['partition']=='test']
    partitions={}
    by_name={row['filename']:row for row in records}
    for name,rows in tables.items():
        selected=[by_name[row['image_name']] for row in rows if row['image_name'] in by_name]
        partitions[name]=dict(rows=len(rows),decoded=len(selected),classes=dict(Counter(row['official_label'] for row in selected)),
                             binary=dict(Counter(row['ground_truth_code'] for row in selected)),
                             patients=len({row['patient_id'] for row in selected}),slides=len({row['slide_id'] for row in selected}))
    integrity=s.patient_partition(train,test)
    overlap={}
    for cohort in ('VAL1','VAL2'):
        manifest=r.read_manifest(Path(f'manifests/{cohort}_manifest.csv'),cohort)
        hashes={row['file_sha256'] for row in manifest}
        matches=[row['sample_id'] for row in records if row['file_sha256'] in hashes]
        overlap[cohort]=dict(count=len(matches),sample_ids=matches,method='JPEG SHA256 versus existing manifest hashes only')
    pairs=s.duplicate_groups(records); pixels=s.duplicate_groups(records,'pixel_sha256')
    selected=s.sample_order(train)
    write_csv(audit_dir/'selected_train.csv',selected)
    summary=dict(dataset=s.DATASET_URL,doi='10.17632/9xxm58dvs3.2',version=2,download_url=s.DOWNLOAD_URL,
        archive_sha256=s.ARCHIVE_SHA,archive_bytes=archive_path.stat().st_size,archive_files=len(entries),
        archive_uncompressed_bytes=sum(v['bytes'] for v in entries),extraction=str(destination),audit_dir=str(audit_dir),
        metadata_sha256={name:hashlib.sha256(value).hexdigest() for name,value in metadata.items()},
        readme=metadata['SICAPv2/readme.txt'].decode(),images=len(image_paths),masks=len(mask_info),decoded_images=len(records),errors=errors,
        image_properties=dict(Counter(json.dumps({k:v for k,v in info.items() if k not in ('info','exif')},sort_keys=True) for info in image_info)),
        image_embedded_metadata=dict(Counter(json.dumps({k:info[k] for k in ('info','exif')},sort_keys=True) for info in image_info)),
        mask_properties=dict(Counter(json.dumps(info,sort_keys=True) for info in mask_info)),
        patients=len({str(row['patient_id']) for row in slides}),slides=len(slides),
        partition_summaries=partitions,patient_partition=integrity,
        official_label_conflicts=conflicts,unlisted=len([v for v in records if v['partition']=='unlisted']),
        mask_pixel_counts={key:sum(v[key] for v in records) for key in [*[f'mask_value_{i}' for i in range(6)],'mask_other']},
        mixed_zero_nonzero=sum(v['mixed_zero_nonzero'] for v in records),multiple_nonzero_values=sum(v['multiple_nonzero_values'] for v in records),
        duplicate_raw_groups=pairs,duplicate_pixel_groups=pixels,overlap=overlap,
        all_official_binary_counts=dict(Counter(v['ground_truth_code'] for v in records if v['ground_truth_code']>=0)),
        selected_binary_counts=dict(Counter(v['ground_truth_code'] for v in selected)),selected_samples=len(selected),
        selected_manifest_sha256=r.sha256_file(audit_dir/'selected_train.csv'),
        selected_manifest=str(audit_dir/'selected_train.csv'),seconds=time.perf_counter()-started,
        VAL2_images_opened=False,training_started=False)
    r.write_json(Path('reports/stage2j_archive_audit.json'),summary)
    print(json.dumps({k:summary[k] for k in ('images','masks','patients','slides','selected_samples','selected_binary_counts','all_official_binary_counts','unlisted','overlap')},indent=2))
    print('errors',len(errors),'raw_duplicate_groups',len(pairs),'pixel_duplicate_groups',len(pixels))


if __name__=='__main__': main()
