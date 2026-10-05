"""Legacy large-tumour-image validation and tumour-area stability sampling."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random
from .gleason import (REFERENCE, REFERENCE_SHA, sha, legacy_score, load_model,
                      gpu_setup, write_csv, write_json, provenance)


def summarize(probabilities):
    import numpy as np
    values = np.asarray(probabilities,dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values)==0 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected nonempty finite nonnegative Nx3 probabilities')
    totals = values.sum(axis=0)
    if totals.sum() <= 0:
        raise ValueError('Zero total probability')
    percentages = totals/totals.sum()*100
    return dict(p_gp3=float(percentages[0]),p_gp4=float(percentages[1]),
                p_gp5=float(percentages[2]),**legacy_score(percentages))


def preprocess_image(image,normalizers):
    import numpy as np
    from PIL import Image
    pixels = np.asarray(image.convert('RGB').resize((350,350),Image.Resampling.LANCZOS))
    brightness,normalizer = normalizers
    result = np.asarray(normalizer.transform(brightness.transform(pixels)),dtype=np.float32)/255.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite normalized pixels')
    return result


def validate(args):
    import numpy as np
    from PIL import Image
    from .evaluate_paper import create_legacy_normalizer
    gpu_setup(42)
    model,meta = load_model(args.model_dir)
    if sha(args.reference) != REFERENCE_SHA:
        raise ValueError('Wrong stain reference')
    normalizers = create_legacy_normalizer(args.reference)
    args.output.mkdir(parents=True,exist_ok=False)
    records, summaries = [],[]
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png','.tif','.tiff'):
            continue
        with Image.open(path) as image:
            width,height=image.size
            nx,ny=width//args.patch_size,height//args.patch_size
            if not nx or not ny:
                raise ValueError(f'Image too small for selected patch size: {path}')
            left=(width-nx*args.patch_size)//2 if args.center_crop else 0
            top=(height-ny*args.patch_size)//2 if args.center_crop else 0
            probabilities=[]
            for y in range(ny):
                for x in range(nx):
                    # Folder 3 uses +1 on all but the first grid row/column;
                    # folder 5 uses a centered crop and aligned coordinates.
                    px=left+x*args.patch_size+(int(x>0) if not args.center_crop else 0)
                    py=top+y*args.patch_size+(int(y>0) if not args.center_crop else 0)
                    tile=image.crop((px,py,px+args.patch_size,py+args.patch_size))
                    prob=model(preprocess_image(tile,normalizers)[None],training=False).numpy()[0]
                    probabilities.append(prob)
                    records.append(dict(image=path.name,row=y,column=x,x=px,y=py,
                                        p_gp3=float(prob[0]),p_gp4=float(prob[1]),p_gp5=float(prob[2])))
            summaries.append(dict(image=path.name,tiles=len(probabilities),**summarize(probabilities)))
    if not records:
        raise ValueError('No images to validate')
    write_csv(args.output/'predictions.csv',records)
    write_csv(args.output/'regions.csv',summaries)
    write_json(args.output/'run.json',dict(provenance(),model_sha256=meta['model_sha256'],
               patch_size=args.patch_size,mpp=args.mpp,fov_um=args.patch_size*args.mpp,
               center_crop=args.center_crop,complete=True))


def stability(args):
    groups=defaultdict(list)
    with args.predictions.open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[row['image']].append([float(row[f'p_gp{k}']) for k in (3,4,5)])
    rng=random.Random(args.seed)
    records=[]
    for name,values in groups.items():
        if len(values)<16:  # same eligibility requirement as released folder 5
            continue
        reference=summarize(values)
        for size in range(1,min(args.max_tiles,len(values))+1):
            for iteration in range(args.rounds):
                # Released script rounds single-patch percentages to one decimal.
                chosen=[[round(v*100,1) for v in row] for row in rng.sample(values,size)]
                score=summarize(chosen)
                records.append(dict(image=name,tiles=size,round=iteration,
                                    area_um2=size*args.fov_um**2,**score,
                                    whole_image_isup=reference['isup'],
                                    same_isup=int(score['isup']==reference['isup'])))
    if not records:
        raise ValueError('No eligible regions with at least 16 tiles')
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'stability.csv',records)
    write_json(args.output/'run.json',dict(provenance(),sampling='without replacement',
               seed=args.seed,rounds=args.rounds,max_tiles=args.max_tiles,fov_um=args.fov_um,
               prediction_sha256=sha(args.predictions),complete=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    v=sub.add_parser('validate')
    v.add_argument('--model-dir',type=Path,required=True)
    v.add_argument('--input',type=Path,required=True)
    v.add_argument('--output',type=Path,required=True)
    v.add_argument('--reference',type=Path,default=REFERENCE)
    v.add_argument('--patch-size',type=int,default=600)
    v.add_argument('--mpp',type=float,required=True,help='Physical pixel spacing of input images')
    v.add_argument('--center-crop',action='store_true',help='Folder 5 centered grid instead of folder 3 offsets')
    s=sub.add_parser('stability')
    s.add_argument('--predictions',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True)
    s.add_argument('--max-tiles',type=int,default=16,help='Paper reports 1-16; use 19 for released loop upper bound')
    s.add_argument('--rounds',type=int,default=20)
    s.add_argument('--fov-um',type=float,required=True)
    s.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.action=='validate':
        if args.patch_size<1 or args.mpp<=0: p.error('Positive patch size and MPP required')
        validate(args)
    else:
        if min(args.max_tiles,args.rounds,args.fov_um)<=0: p.error('Positive sampling parameters required')
        stability(args)


if __name__=='__main__':
    main()
