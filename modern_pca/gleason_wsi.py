"""Runtime adaptation of the released C1/C8/environment/Type-2 WSI workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
from .gleason import (REFERENCE,REFERENCE_SHA,sha,load_model,gpu_setup,
                      legacy_score,write_csv,write_json,provenance)
from .gleason_regions import preprocess_image


def eight_views(image):
    from PIL import ImageOps
    r90,r180,r270=[image.rotate(a) for a in (90,180,270)]
    return [image,r90,r180,r270,ImageOps.flip(r90),ImageOps.flip(r270),
            ImageOps.flip(image),ImageOps.mirror(image)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slide',type=Path,required=True)
    p.add_argument('--tumour-model',type=Path,required=True)
    p.add_argument('--tumour-classes',type=int,choices=[2,3],default=3)
    p.add_argument('--tumour-index',type=int,default=2)
    p.add_argument('--gleason-model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--patch-size',type=int,default=600,help='Legacy level-0 pixels; explicitly select for other resolutions')
    p.add_argument('--c8',action='store_true')
    p.add_argument('--environment',action='store_true')
    args=p.parse_args()
    if args.patch_size<1 or not 0<=args.tumour_index<args.tumour_classes:
        p.error('Invalid patch size or tumour index')
    tf=gpu_setup(42)
    import numpy as np
    from PIL import Image
    import openslide
    from .evaluate_paper import create_legacy_normalizer
    if sha(args.reference)!=REFERENCE_SHA:
        raise ValueError('Wrong stain reference')
    normalizers=create_legacy_normalizer(args.reference)
    tumour=tf.keras.models.load_model(args.tumour_model,compile=False)
    if tumour.output_shape!=(None,args.tumour_classes) or tumour.input_shape!=(None,350,350,3):
        raise ValueError('Tumour model must accept normalized 350 RGB and requested class count')
    args.output.mkdir(parents=True,exist_ok=False)
    slide=openslide.OpenSlide(str(args.slide))
    try:
        width,height=slide.dimensions
        size=args.patch_size
        nx,ny=width//size,height//size
        if not nx or not ny: raise ValueError('Slide smaller than one tile')
        probabilities=np.zeros((ny,nx,args.tumour_classes),dtype=np.float32)
        def raw(x,y):
            return slide.read_region((x,y),0,(size,size)).convert('RGB')
        for y in range(ny):
            for x in range(nx):
                # Preserve folder 4 detection coordinates, including +1 offsets.
                patch=raw(x*size+int(x>0),y*size+int(y>0))
                resized=np.asarray(patch.resize((350,350),Image.Resampling.LANCZOS))
                blue=resized[:,:,2]
                if np.count_nonzero((blue>100)&(blue<200))<=7000 or np.count_nonzero(blue<100)<=10:
                    continue
                pixels=preprocess_image(patch,normalizers)
                prob=tumour(pixels[None],training=False).numpy()[0]
                if args.c8 and .2<=prob[args.tumour_index]<.8:
                    normalized=Image.fromarray(np.rint(pixels*255).astype(np.uint8))
                    views=np.stack([np.asarray(im,dtype=np.float32)/255. for im in eight_views(normalized)])
                    prob=np.median(tumour(views,training=False).numpy(),axis=0)
                probabilities[y,x]=prob
        tissue=probabilities.sum(axis=2)>0
        positive=probabilities[:,:,args.tumour_index]>.5
        if args.environment:
            # Preserve sequential updates and four-neighbour rule; explicit
            # bounds fix legacy negative wraparound/out-of-range indexing.
            for y in range(ny):
                for x in range(nx):
                    if not positive[y,x]: continue
                    adjacent=any(0<=yy<ny and 0<=xx<nx and positive[yy,xx]
                                 for yy,xx in ((y-1,x),(y,x+1),(y+1,x),(y,x-1)))
                    if adjacent: continue
                    cx,cy=x*size+size//2,y*size+size//2
                    # Legacy environment path performs resize + /255 only.
                    offsets=((cx-size,cy-size),(cx,cy-size),(cx,cy),(cx-size,cy))
                    patches=np.stack([np.asarray(raw(px,py).resize((350,350),Image.Resampling.LANCZOS),
                                                       dtype=np.float32)/255. for px,py in offsets])
                    positive[y,x]=np.max(tumour(patches,training=False).numpy()[:,args.tumour_index])>=.5
        del tumour
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        model,meta=load_model(args.gleason_model_dir)
        gp=np.zeros((ny,nx,3),dtype=np.float32)
        records=[]
        counters=np.array([.01,.01,.01]) # legacy pseudocount retained for nonempty slides
        isup_map=np.zeros((ny,nx),dtype=np.uint8)
        for y,x in np.argwhere(positive):
            prob=model(preprocess_image(raw(int(x)*size,int(y)*size),normalizers)[None],training=False).numpy()[0]
            gp[y,x]=prob
            score=[round(float(v)*100,1) for v in prob]
            counters+=score
            result=legacy_score(score)
            isup_map[y,x]=result['isup']
            records.append(dict(image=args.slide.name,row=int(y),column=int(x),
                                p_gp3=float(prob[0]),p_gp4=float(prob[1]),p_gp5=float(prob[2]),**result))
        if records: write_csv(args.output/'predictions.csv',records)
        else: write_csv(args.output/'predictions.csv',[],['image','row','column','p_gp3','p_gp4','p_gp5'])
        np.savez_compressed(args.output/'maps.npz',tumour_probabilities=probabilities,
                            tissue=tissue,tumour=positive,gleason_probabilities=gp,isup=isup_map)
        palette=np.array([[255,255,255],[30,170,70],[180,210,40],[255,190,0],[255,90,0],[180,0,0]],dtype=np.uint8)
        heat=palette[isup_map]
        heat[tissue&~positive]=[130,160,230]
        heatmap=Image.fromarray(heat).resize((nx*10,ny*10),Image.Resampling.NEAREST)
        heatmap.save(args.output/'heatmap.png')
        thumb=slide.get_thumbnail((1600,1600)).convert('RGB')
        overlay=Image.new('RGB',thumb.size,'white')
        covered=(max(1,round(nx*size/width*thumb.width)),max(1,round(ny*size/height*thumb.height)))
        overlay.paste(heatmap.resize(covered,Image.Resampling.NEAREST),(0,0))
        Image.blend(thumb,overlay,.2).save(args.output/'overlay.png')
        write_json(args.output/'run.json',dict(provenance(),gleason_model_sha256=meta['model_sha256'],
                   tumour_model_sha256=sha(args.tumour_model),tumour_tiles=len(records),
                   pattern_percentages=(counters/counters.sum()*100).tolist() if records else None,
                   patch_size=size,mpp_x=slide.properties.get('openslide.mpp-x'),
                   mpp_y=slide.properties.get('openslide.mpp-y'),c8=args.c8,environment=args.environment,
                   complete=True))
    finally:
        slide.close()


if __name__=='__main__':
    main()
