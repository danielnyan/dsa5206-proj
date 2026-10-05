from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image
from openpyxl import Workbook

from modern_pca.gleason import CLASSES,SCHEDULE,legacy_score,sha,read_rows,write_json,write_csv
from modern_pca.prepare_gleason import crowd_labels,sicap_labels,audit_splits,extract,image_index,tumour_rows
from modern_pca.gleason_regions import summarize,stability
from modern_pca.gleason_wsi import eight_views


def test_crowd_uses_supplied_vote_not_new_tie_rule(tmp_path):
    for split in ('train','val','test'):
        key='ground truth' if split=='test' else 'MV'
        row={f'marker{i}':v for i,v in enumerate([0,2,-1,-1,-1,-1,-1],1)}
        row.update({'Patch filename':'case_row_0_column_0',key:2 if split!='test' else 1})
        write_csv(tmp_path/f'{split}.csv',[row])
    rows=list(crowd_labels(tmp_path))
    assert [r['source_label'] for r in rows]==[2,2,1]
    assert [r['split'] for r in rows]==['train','validation','test']
    assert rows[0]['source_wsi']=='case'


def test_sicap_mapping_and_cribriform_is_not_class(tmp_path):
    for name in ('Train','Val','Test'):
        wb=Workbook()
        sheet=wb.active
        sheet.append(['image_name','NC','G3','G4','G5','G4C'])
        sheet.append(['slide_Block_Region_1.jpg',0,0,1,0,1])
        wb.save(tmp_path/f'{name}.xlsx')
    assert [r['source_label'] for r in sicap_labels(tmp_path)]==[2,2,2]
    assert all(r['source_g4c']==1 for r in sicap_labels(tmp_path))
    wb=Workbook(); sheet=wb.active
    sheet.append(['image_name','NC','G3','G4','G5'])
    sheet.append(['bad.jpg',0,1,1,0]); wb.save(tmp_path/'Train.xlsx')
    with pytest.raises(ValueError,match='one-hot'): list(sicap_labels(tmp_path))


@pytest.mark.parametrize('dataset',['crowd','sicap'])
def test_only_nc_removed_and_classes_remapped(dataset):
    records=[dict(dataset=dataset,source_label=k,votes='[-1, -1, 1]') for k in range(4)]
    result=tumour_rows(records)
    assert [r['source_label'] for r in result]==[1,2,3]
    assert [r['label'] for r in result]==[0,1,2]
    assert [r['class_name'] for r in result]==CLASSES
    assert all(r['votes']=='[-1, -1, 1]' for r in result)


def test_masks_not_indexed(tmp_path):
    (tmp_path/'images').mkdir(); (tmp_path/'masks').mkdir()
    for folder in ('images','masks'):
        Image.new('RGB',(4,4)).save(tmp_path/f'{folder}/a.jpg')
    assert image_index(tmp_path)=={'a':tmp_path/'images/a.jpg'}


def test_split_audit_reports_slide_overlap_and_rejects_content_leak():
    rows=[dict(dataset='crowd',name='a',source_wsi='same',split='train',file_sha256='1'),
          dict(dataset='crowd',name='b',source_wsi='same',split='validation',file_sha256='2')]
    assert len(audit_splits(rows)['slide_overlaps'])==1
    rows[1]['file_sha256']='1'
    with pytest.raises(ValueError,match='Identical'): audit_splits(rows)


def test_zip_traversal_rejected(tmp_path):
    path=tmp_path/'other.zip'
    with zipfile.ZipFile(path,'w') as z: z.writestr('../outside','bad')
    with pytest.raises(ValueError,match='Unsafe'): extract(path,tmp_path/'extracted')
    assert not (tmp_path/'outside').exists()


@pytest.mark.parametrize('values,expected',[
    ([96,4,0],(3,3,1)),([70,30,0],(3,4,2)),([30,70,0],(4,3,3)),
    ([0,92,8],(4,5,5)),([95,5,0],(3,3,1)),([94.9,5.1,0],(3,4,2)),
    ([90,5,5],(3,3,1)),([89.9,5,5.1],(3,5,4)),([50,50,0],(4,3,3))])
def test_legacy_scoring(values,expected):
    result=legacy_score(values)
    assert (result['primary'],result['secondary'],result['isup'])==expected


def test_soft_aggregation_does_not_argmax():
    result=summarize([[.6,.4,0],[.7,.3,0]])
    assert result['p_gp4']==pytest.approx(35)
    assert result['isup']==2


def test_schedule_and_propagated_batchnorm():
    assert sum(s[1] for s in SCHEDULE)==14
    assert [s[0].split('_')[-1] for s in SCHEDULE]==['17','14','11','9','7','5','3','1']
    source=(Path(__file__).parents[1]/'modern_pca/train_gleason.py').read_text()
    ast.parse(source)
    assert 'backbone_training=None' in source


def test_eight_unique_d4_views():
    image=Image.fromarray(np.arange(27,dtype=np.uint8).reshape(3,3,3))
    assert len({np.asarray(v).tobytes() for v in eight_views(image)})==8


def test_stability_bounds_sample_count(tmp_path):
    from argparse import Namespace
    records=[dict(image='region',p_gp3=.7,p_gp4=.3,p_gp5=0) for _ in range(16)]
    path=tmp_path/'predictions.csv'; write_csv(path,records)
    stability(Namespace(predictions=path,output=tmp_path/'out',seed=42,
                        rounds=2,max_tiles=19,fov_um=150))
    with (tmp_path/'out/stability.csv').open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==32 and max(int(r['tiles']) for r in rows)==16
    assert all(r['same_isup']=='1' for r in rows)


def test_manifest_integrity(tmp_path):
    root=tmp_path/'manifests'; root.mkdir()
    path=root/'crowd_train.csv'
    row=dict(dataset='crowd',split='train',label=0,class_name=CLASSES[0])
    write_csv(path,[row]); write_json(tmp_path/'prepared.json',dict(manifests={path.name:sha(path)}))
    assert read_rows(root,'crowd','train')[0]['label']==0
    path.write_text('tampered')
    with pytest.raises(ValueError,match='checksum'): read_rows(root,'crowd','train')


def test_real_normalization_cache_matches_existing_adapter(tmp_path):
    pytest.importorskip('staintools')
    from modern_pca.prepare_gleason import init_worker,prepare_image
    from modern_pca.gleason import REFERENCE,ROOT
    from modern_pca.evaluate_paper import create_legacy_normalizer,preprocess_legacy
    source=next((ROOT/'1_training/Examples/training_gleason_grading/gp3').glob('*.jpg'))
    row=dict(sample_id='test:a',file_sha256=sha(source),path=str(source))
    init_worker(str(REFERENCE),str(tmp_path))
    cached=prepare_image(row)
    values=np.asarray(Image.open(tmp_path/cached['cache_file']),dtype=np.float32)/255
    expected=preprocess_legacy(row,source,create_legacy_normalizer(REFERENCE))
    np.testing.assert_array_equal(values,expected)
