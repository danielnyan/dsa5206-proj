"""Stage 2E: bounded, lossless VAL1-training cache experiments only.

The builder has a hard 1,024-sample limit. It never generates a full cohort cache
or changes frozen splits. Cache uint8 after Macenko, before scaling/augmentation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import reimplementation as r

SHAPE = (350,350,3)
SAMPLE_BYTES = int(np.prod(SHAPE))
SPLIT_HASHES = {
    "VAL1_train.csv": "e6aaa451d54602fa19c86ea37b395043ed8fb0526122d6e780a1ecd410ebc3b8",
    "VAL1_internal_validation.csv": "cdcecc2362f175c4aa6163c2f6304071f11624bb1b57f4d0aef9f31e71055069",
}


def frozen_training_rows(directory=Path("manifests/VAL1_internal_v1")):
    """Read/check both split CSV bytes, but only parse training sample metadata."""
    for filename, digest in SPLIT_HASHES.items():
        r.sha256_file(Path(directory)/filename,digest)
    rows=list(csv.DictReader((Path(directory)/"VAL1_train.csv").open(newline="",encoding="utf-8")))
    for row in rows:
        for key in ("row_index","ground_truth_code","file_size_bytes","width","height"):
            row[key]=int(row[key])
    r.validate_rows(rows,"VAL1",r.TRAIN_COUNTS)
    return rows


def representative_rows(rows, count=1000):
    if not 100 <= count <= 1024 or any(x["cohort"]!="VAL1" for x in rows):
        raise ValueError("Only bounded VAL1-training subsets (100..1024)")
    # Preserve the training class ratio, while explicitly including dimension strata.
    counts=Counter(x["class_name"] for x in rows)
    benign=round(count*counts["benign"]/len(rows))
    selections={}
    for label,n in (("benign",benign),("tumour",count-benign)):
        ranked=sorted((x for x in rows if x["class_name"]==label),key=lambda x:r.rank_id(x["sample_id"],"VAL1-stage2e-v1|42|"))
        strata={}
        for row in ranked:
            strata.setdefault((row["width"],row["height"]),row)
        mandatory=list(strata.values())
        if len(mandatory)>n:
            raise ValueError("Subset too small for native-dimension strata")
        ids={x["sample_id"] for x in mandatory}
        selections[label]=mandatory+[x for x in ranked if x["sample_id"] not in ids][:n-len(mandatory)]
    # Spread classes through each logical batch; put all dimension strata early.
    result=[]
    while any(selections.values()):
        for label in ("benign","tumour"):
            if selections[label]:
                result.append(selections[label].pop(0))
    if len(result)!=count or len({x["sample_id"] for x in result})!=count:
        raise ValueError("Subset identity mismatch")
    return result


def normalized_uint8(row,path,normalizers,timings=None):
    """Instrument the exact strict-evaluator operations, without modifying it."""
    timings=defaultdict(float) if timings is None else timings
    before=path.stat()
    t=time.perf_counter(); payload=path.read_bytes(); timings["file_read"]+=time.perf_counter()-t
    t=time.perf_counter()
    if hashlib.sha256(payload).hexdigest()!=row["file_sha256"]:
        raise ValueError("Source JPEG checksum changed")
    timings["integrity"]+=time.perf_counter()-t
    t=time.perf_counter()
    with Image.open(io.BytesIO(payload)) as image:
        rgb=image.convert("RGB")
    timings["decode_rgb"]+=time.perf_counter()-t
    t=time.perf_counter(); resized=rgb.resize((350,350),Image.Resampling.LANCZOS); pixels=np.array(resized); timings["resize"]+=time.perf_counter()-t
    t=time.perf_counter(); pixels=normalizers[0].transform(pixels); timings["brightness"]+=time.perf_counter()-t
    t=time.perf_counter(); pixels=normalizers[1].transform(pixels); timings["macenko"]+=time.perf_counter()-t
    if pixels.dtype!=np.uint8 or pixels.shape!=SHAPE:
        raise ValueError("Macenko did not return native uint8 RGB; never quantize implicitly")
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("Source changed during read")
    return pixels


def to_float(pixels):
    if pixels.dtype!=np.uint8 or pixels.shape[-3:]!=SHAPE:
        raise ValueError("Expected uint8 350x350x3 cache")
    return np.asarray(pixels,dtype=np.float32)/np.float32(255.0)


def exact_comparison(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(pass_=False,reason="shape/dtype mismatch")
    delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(pass_=bool(np.array_equal(a,b)),max_absolute_difference=float(delta.max(initial=0)),
                differing_elements=int(np.count_nonzero(a!=b)))


def cache_provenance():
    sources="\n".join(inspect.getsource(f) for f in (normalized_uint8,to_float,r.preprocess_legacy,r.create_legacy_normalizer))
    return dict(source_VAL1_manifest_sha256=r.VAL1_SHA,split_sha256=SPLIT_HASHES,
                stain_reference_sha256=r.REFERENCE_SHA,preprocessing_implementation_sha256=hashlib.sha256(sources.encode()).hexdigest(),
                environment=r.capture_environment(),settings=dict(r.preprocessing_settings(),resize="Pillow LANCZOS 350x350",dtype="uint8",shape=SHAPE,
                                                                cache_position="after Macenko; before float32/255 and augmentation"))


def item_metadata(row,shard,index,pixels):
    return dict(sample_id=row["sample_id"],relative_path=row["relative_path"],source_jpeg_sha256=row["file_sha256"],
                tensor_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),shape=list(SHAPE),dtype="uint8",
                class_name=row["class_name"],ground_truth_code=row["ground_truth_code"],cohort="VAL1",split="train",
                shard=shard,index=index)


def build_subset_cache(rows,extracted,output,reference=r.REFERENCE,shard_size=256,component_observer=None):
    if not rows or len(rows)>1024 or shard_size<1:
        raise ValueError("Subset cache only; full-cohort generation is not authorized")
    train_ids={x["sample_id"] for x in frozen_training_rows()}
    if len({x["sample_id"] for x in rows})!=len(rows) or any(x["sample_id"] not in train_ids for x in rows):
        raise ValueError("Cache rows must be unique frozen VAL1-training members")
    output=Path(output); r.ensure_output_separation(extracted,[output]); output.mkdir(parents=True,exist_ok=True)
    provenance=cache_provenance(); context=dict(provenance=provenance,sample_ids=[x["sample_id"] for x in rows],shard_size=shard_size)
    context_hash=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    if (output/"context.json").exists():
        if json.loads((output/"context.json").read_text())["fingerprint"]!=context_hash:
            raise ValueError("Cache context mismatch; refuse stale resume")
    else:
        r.write_json(output/"context.json",dict(fingerprint=context_hash,context=context,created_at_utc=r.utc_now()))
    normalizers=r.normalizers(reference); timings=defaultdict(float); all_items=[]; start=time.perf_counter()
    for offset in range(0,len(rows),shard_size):
        name=f"shard_{offset//shard_size:04d}.npy"; sidecar=output/(name+".json")
        batch=rows[offset:offset+shard_size]
        if sidecar.exists():
            committed=json.loads(sidecar.read_text())
            if committed["context_fingerprint"]!=context_hash or [x["sample_id"] for x in committed["items"]]!=[x["sample_id"] for x in batch]:
                raise ValueError("Committed shard mapping mismatch")
            r.sha256_file(output/name,committed["file_sha256"])
            all_items.extend(committed["items"]); continue
        arrays=[]; items=[]
        for index,row in enumerate(batch):
            pixels=normalized_uint8(row,r.image_path(row,extracted),normalizers,timings)
            if component_observer is not None:
                component_observer(pixels,row,timings)
            arrays.append(pixels); items.append(item_metadata(row,name,index,pixels))
            if (offset+index+1)%50==0:
                r.LOG.info("Cache %d/%d",offset+index+1,len(rows))
        temporary=output/(name+".tmp")
        with temporary.open("wb") as stream:
            np.save(stream,np.stack(arrays),allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(output/name)
        r.write_json(sidecar,dict(context_fingerprint=context_hash,file_sha256=r.sha256_file(output/name),items=items))
        all_items.extend(items)
    meta=dict(status="COMPLETE",created_at_utc=r.utc_now(),provenance=provenance,context_fingerprint=context_hash,
              items=all_items,samples=len(rows),shard_size=shard_size,generation_seconds=time.perf_counter()-start,
              components_seconds=dict(timings),source_images_modified=False,VAL2_images_accessed=False)
    r.write_json(output/"cache.json",meta)
    return meta


class CacheReader:
    """Read-only mmap handles; construct a fresh reader in each process."""
    def __init__(self,root,verify=True):
        self.root=Path(root); self.meta=json.loads((self.root/"cache.json").read_text())
        if self.meta["status"]!="COMPLETE":
            raise ValueError("Incomplete cache")
        self.items=self.meta["items"]; self.arrays={}; ids=set()
        sources={row['sample_id']:row for row in frozen_training_rows()}
        for item in self.items:
            if item["sample_id"] in ids or item["cohort"]!="VAL1" or item["split"]!="train" or item["dtype"]!="uint8" or item["shape"]!=list(SHAPE):
                raise ValueError("Invalid cache identity/contract")
            if item["class_name"] not in r.CLASSES or item["ground_truth_code"]!=r.CLASSES[item["class_name"]]:
                raise ValueError("Invalid cache class mapping")
            source=sources.get(item['sample_id'])
            if source is None or any(item[a]!=source[b] for a,b in [('relative_path','relative_path'),('class_name','class_name'),
                    ('ground_truth_code','ground_truth_code'),('source_jpeg_sha256','file_sha256')]):
                raise ValueError('Cache/sample manifest mapping mismatch')
            ids.add(item["sample_id"])
            path=self.root/item["shard"]
            if path.parent!=self.root or path.is_symlink():
                raise ValueError("Unsafe cache shard")
            if item["shard"] not in self.arrays:
                array=np.load(path,mmap_mode="r",allow_pickle=False)
                if array.dtype!=np.uint8 or array.shape[1:]!=SHAPE:
                    raise ValueError("Invalid shard dtype/shape")
                self.arrays[item["shard"]]=array
            if verify and hashlib.sha256(self.get(len(ids)-1).tobytes()).hexdigest()!=item["tensor_sha256"]:
                raise ValueError("Cached tensor checksum mismatch")
    def get(self,index):
        item=self.items[int(index)]
        return np.array(self.arrays[item["shard"]][item["index"]],copy=True)


def epoch_order(items,epoch=1):
    return sorted(range(len(items)),key=lambda i:r.rank_id(items[i]["sample_id"],f"VAL1-epoch-v1|42|{epoch}|"))


def dataset(reader,indices,batch_size=4,parallel=1,prefetch=0,device_prefetch=False,augment=True,epoch=1):
    import tensorflow as tf
    labels=np.array([reader.items[i]["ground_truth_code"] for i in indices],np.int64)
    seeds=np.stack([r.flip_seed(reader.items[i]["sample_id"],epoch) for i in indices])
    with tf.device("/CPU:0"):
        ds=tf.data.Dataset.from_tensor_slices((np.array(indices,np.int64),labels,seeds))
        def read(index,label,seed):
            # Preserve NumPy float32 division exactly; do not permit graph constant
            # folding to substitute a rounded reciprocal multiplication.
            image=tf.numpy_function(lambda i:to_float(reader.get(i)),[index],tf.float32,stateful=False)
            image.set_shape(SHAPE)
            if augment:
                decisions=tf.random.stateless_uniform([2],seed)<.5
                image=tf.cond(decisions[0],lambda:tf.reverse(image,[1]),lambda:image)
                image=tf.cond(decisions[1],lambda:tf.reverse(image,[0]),lambda:image)
            return image,label,index
        options=tf.data.Options(); options.deterministic=True
        options.threading.private_threadpool_size=max(2,parallel)
        ds=ds.with_options(options).map(read,num_parallel_calls=parallel,deterministic=True).batch(batch_size,drop_remainder=False)
        if prefetch:
            ds=ds.prefetch(prefetch)
    if device_prefetch:
        ds=ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0",buffer_size=1))
    return ds


def storage_requirement(counts=(116655,29164),shard_size=256):
    """Exact NPY v1 payload+header bytes; metadata reported separately."""
    size=0; shards=0
    for count in counts:
        for offset in range(0,count,shard_size):
            n=min(shard_size,count-offset); stream=io.BytesIO()
            np.lib.format.write_array_header_1_0(stream,dict(descr="|u1",fortran_order=False,shape=(n,*SHAPE)))
            size+=n*SAMPLE_BYTES+len(stream.getvalue()); shards+=1
    return dict(images=sum(counts),raw_tensor_bytes=sum(counts)*SAMPLE_BYTES,npy_bytes=size,shards=shards,
                metadata_scope="Excludes JSON metadata, filesystem allocation units and temporary generation workspace")
