from pathlib import Path

p = Path("/scratch/e1536052/DSA5206/singapore_external/scripts/evaluate_val2_singapore15.py")
s = p.read_text()

s = s.replace(
    "from . import reimplementation as r",
    "from modern_pca import reimplementation as r"
)
s = s.replace(
    "from .evaluate_paper import",
    "from modern_pca.evaluate_paper import"
)

s = s.replace(
    "p.add_argument('--checkpoint-manifest', type=Path, required=True)",
    "p.add_argument('--training-summary', type=Path, required=True)"
)
s = s.replace(
    "    p.add_argument('--final-model-metadata', type=Path, required=True)\n",
    ""
)

s = s.replace(
    "args.checkpoint_manifest, args.final_model_metadata,",
    "args.training_summary,"
)

start = s.index("    check = json.loads(args.checkpoint_manifest.read_text())")
end = s.index("    import tensorflow as tf", start)

replacement = '''
    summary = json.loads(args.training_summary.read_text())
    model_digest = sha256(args.model)

    if summary.get("checkpoint_sha256") != model_digest:
        raise RuntimeError("Singapore epoch-15 model SHA256 mismatch")
    if summary.get("samples") != 11020:
        raise RuntimeError("Unexpected Singapore training count")
    if summary.get("updates") != 111:
        raise RuntimeError("Unexpected training update count")
    if summary.get("source_optimizer_iterations") != 1167:
        raise RuntimeError("Unexpected starting Adam iterations")
    if summary.get("final_optimizer_iterations") != 1278:
        raise RuntimeError("Unexpected final Adam iterations")
    if summary.get("learning_rate") != 1e-6:
        raise RuntimeError("Unexpected learning rate")
    if summary.get("trainable_boundary") != "normal_conv_1_1":
        raise RuntimeError("Unexpected fine-tuning boundary")

    print("Singapore epoch-15 model verified:", model_digest, flush=True)

'''

s = s[:start] + replacement + s[end:]

s = s.replace(
    "'purpose':'vanda_epoch15_SICAPv2_independent_VAL2_binary_evaluation'",
    "'purpose':'vanda_epoch15_Singapore_all11020_VAL2_binary_evaluation'"
)

s = s.replace(
    "'checkpoint_manifest_sha256':sha256(args.checkpoint_manifest),",
    "'training_summary_sha256':sha256(args.training_summary),"
)
s = s.replace(
    "'final_model_metadata_sha256':sha256(args.final_model_metadata),",
    "'training_data':'Singapore 11020 eligible gland patches',"
)

assert "args.checkpoint_manifest" not in s
assert "args.final_model_metadata" not in s
assert "vanda_epoch15_SICAPv2_independent_VAL2_binary_evaluation" not in s

p.write_text(s)
print("PASS: Singapore VAL2 evaluator prepared")
