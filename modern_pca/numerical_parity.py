"""Predeclared CPU/GPU tolerances and bounded-memory comparison utilities."""
from __future__ import annotations
import numpy as np

TOLERANCES={
    "forward_probability_atol":1e-5,"forward_probability_rtol":1e-4,
    "forward_logits_atol":1e-4,"forward_logits_rtol":1e-4,"forward_loss_atol":1e-4,
    "gradient_max_atol":1e-4,"gradient_max_rtol":1e-3,
    "gradient_mean_atol":1e-6,"gradient_mean_rtol":1e-4,"gradient_relative_l2":1e-3,
    "weight_max_atol_per_update":5e-5,"weight_mean_atol":1e-6,
    "post_probability_atol":1e-4,"post_probability_rtol":1e-3,"post_loss_atol":1e-3,
    "class_agreement":1.0,"relative_denominator_floor":1e-8,
}
RATIONALE=("Float32 cross-device kernel/reduction differences are allowed, never bitwise identity. "
           "Forward tolerances are much smaller than the 0.5 decision threshold; class agreement is separately mandatory. "
           "Gradients use per-tensor absolute, relative-scale and relative-L2 checks; near-zero tensors use absolute floors. "
           "Adam near-zero gradients can amplify small reduction differences, so a per-update 5e-5 maximum weight bound "
           "is paired with a 1e-6 mean bound and post-update probability/loss checks. These are diagnostic acceptance "
           "criteria, not proof of equivalent 14-epoch trajectories. Declared before any CPU/GPU observations.")


def difference(a,b,chunk=1000000):
    if a.shape!=b.shape:
        raise ValueError("Comparison shape mismatch")
    aa=a.reshape(-1); bb=b.reshape(-1); maximum=relative=total=squared=norm_a=norm_b=max_a=abs_a=0.
    for offset in range(0,len(aa),chunk):
        x=np.asarray(aa[offset:offset+chunk],np.float64); y=np.asarray(bb[offset:offset+chunk],np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite comparison")
        d=np.abs(x-y); maximum=max(maximum,float(d.max(initial=0)))
        relative=max(relative,float((d/np.maximum(np.abs(x),TOLERANCES["relative_denominator_floor"])).max(initial=0)))
        total+=float(d.sum()); squared+=float(np.dot(d,d)); norm_a+=float(np.dot(x,x)); norm_b+=float(np.dot(y,y))
        max_a=max(max_a,float(np.abs(x).max(initial=0))); abs_a+=float(np.abs(x).sum())
    return dict(elements=len(aa),max_absolute=maximum,mean_absolute=total/max(1,len(aa)),max_relative=relative,
                rms=float(np.sqrt(squared/max(1,len(aa)))),relative_l2=float(np.sqrt(squared)/max(np.sqrt(norm_a),1e-8)),
                reference_max_absolute=max_a,reference_mean_absolute=abs_a/max(1,len(aa)),reference_l2=float(np.sqrt(norm_a)))


def gradient_pass(stats):
    t=TOLERANCES
    return (stats["max_absolute"]<=t["gradient_max_atol"]+t["gradient_max_rtol"]*stats["reference_max_absolute"] and
            stats["mean_absolute"]<=t["gradient_mean_atol"]+t["gradient_mean_rtol"]*stats["reference_mean_absolute"] and
            (stats["relative_l2"]<=t["gradient_relative_l2"] or stats["reference_l2"]<1e-6))


def weights_pass(stats,updates):
    return stats["max_absolute"]<=TOLERANCES["weight_max_atol_per_update"]*updates and stats["mean_absolute"]<=TOLERANCES["weight_mean_atol"]


def forward_comparison(cpu,gpu,post=False):
    t=TOLERANCES; prefix="post" if post else "forward"
    a=np.asarray(cpu["probabilities"]); b=np.asarray(gpu["probabilities"])
    result=dict(probabilities=difference(a,b),loss_difference=abs(cpu["loss"]-gpu["loss"]),
                predicted_class_agreement=float(np.mean((a[:,1]>.5)==(b[:,1]>.5))))
    result["pass"]=bool(np.allclose(a,b,atol=t[prefix+"_probability_atol"],rtol=t[prefix+"_probability_rtol"]) and
                        result["loss_difference"]<=t[prefix+"_loss_atol"] and result["predicted_class_agreement"]==1.)
    if not post:
        result["logits"]=difference(np.asarray(cpu["logits"]),np.asarray(gpu["logits"]))
        result["pass"] &= bool(np.allclose(cpu["logits"],gpu["logits"],atol=t["forward_logits_atol"],rtol=t["forward_logits_rtol"]))
    return result
