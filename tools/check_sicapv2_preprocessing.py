"""Read-only preprocessing assessment of official SICAP TRAIN images; no TensorFlow."""
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from modern_pca import reimplementation as r, sicapv2 as s
from tools.audit_sicapv2 import guard
from tools.inspect_sicapv2 import ROOT


def main():
    guard()
    # Images are extracted before masks; this independent check may overlap the
    # archive audit, using the same authenticated official training spreadsheet.
    with zipfile.ZipFile(ROOT/'SICAPv2.zip') as archive:
        official=s.table(archive.read('SICAPv2/partition/Test/Train.xlsx'),['image_name','NC','G3','G4','G5','G4C'])
    rows=[]
    for entry in official:
        label,binary=s.primary_label(entry)
        path=ROOT/'extracted/SICAPv2/images'/entry['image_name']
        rows.append(dict(sample_id='SICAPv2:'+entry['image_name'],filename=entry['image_name'],
            official_label=label,ground_truth_code=binary,file_sha256=r.sha256_file(path)))
    rows=s.sample_order(rows)
    reference = r.normalizers(r.REFERENCE)
    failures = []
    records = []
    panels = []
    started = time.perf_counter()
    # Stratified plus brightest/lowest contrast examples, chosen without outcomes.
    visual_ids = set()
    for label in ('NC', 'G3', 'G4', 'G5'):
        # Exactly the first two per official grade in the frozen order.
        visual_ids.update(row['sample_id'] for row in [v for v in rows if v['official_label']==label][:2])
    for index,row in enumerate(rows):
        path = ROOT/'extracted/SICAPv2/images'/row['filename']
        tick = time.perf_counter()
        try:
            image = r.preprocess_legacy(row,path,reference)
            records.append(dict(sample_id=row['sample_id'],seconds=time.perf_counter()-tick,
                                minimum=float(image.min()),maximum=float(image.max()),
                                mean_rgb=image.mean(axis=(0,1)).tolist(),std=float(image.std())))
            if row['sample_id'] in visual_ids:
                raw = Image.open(path).convert('RGB').resize((175,175))
                normalized = Image.fromarray(np.rint(image*255).astype(np.uint8)).resize((175,175))
                panels.append((row['sample_id'],raw,normalized))
        except Exception as error:
            failures.append(dict(sample_id=row['sample_id'],error=str(error)))
        if (index+1)%100==0:
            print('preprocessed',index+1,'failures',len(failures),'seconds',time.perf_counter()-started,flush=True)
    canvas = Image.new('RGB',(350,195*len(panels)),'white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for index,(name,raw,normalized) in enumerate(panels):
        canvas.paste(raw,(0,index*195)); canvas.paste(normalized,(175,index*195))
        draw.text((0,index*195+176),name[:52],fill='black')
    canvas.save('reports/stage2j_preprocessing_contact_sheet.png')
    report = dict(samples=len(rows),successes=len(records),failures=failures,
        failure_rate=len(failures)/len(rows),reference_sha256=r.REFERENCE_SHA,
        selected_image_sha256={row['sample_id']:row['file_sha256'] for row in rows},
        seconds=time.perf_counter()-started,records=records,visual_samples=[p[0] for p in panels],
        training_started=False,VAL2_images_opened=False,cache_built=False)
    r.write_json(Path('reports/stage2j_preprocessing.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','visual_samples','selected_image_sha256')},indent=2))


if __name__=='__main__': main()
