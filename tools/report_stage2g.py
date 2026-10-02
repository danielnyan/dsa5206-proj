"""Report Stage 2G only from completed generation, verification and readiness runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modern_pca import full_cache as f
from modern_pca import reimplementation as r


def forecast(previous, loader, checkpoint_seconds):
    """Measured full-input floor; compute remains Stage 2E's bounded estimate."""
    stages = []
    for stage in previous['stages']:
        compute_train = 116655 / stage['train_images_per_second']
        compute_val = 29164 / stage['validation_images_per_second']
        input_train = 116655 / loader['splits']['train']['images_per_second']
        input_val = 29164 / loader['splits']['internal_validation']['images_per_second']
        stages.append(dict(boundary=stage['boundary'], epochs=stage['epochs'],
                           overlap_seconds_per_epoch=max(compute_train, input_train) + max(compute_val, input_val),
                           conservative_serial_seconds_per_epoch=compute_train + compute_val + input_train + input_val))
    # New resumable trainer retains a checkpoint after each epoch, in addition
    # to the original final portable model. Include that operational overhead.
    added_checkpoint = 14 * checkpoint_seconds
    common = previous['checkpoint_io_seconds'] + previous['startup_compilation_seconds']
    return dict(epochs_14_seconds=sum(x['epochs'] * x['overlap_seconds_per_epoch'] for x in stages) + common + added_checkpoint,
                conservative_serial_seconds=sum(x['epochs'] * x['conservative_serial_seconds_per_epoch'] for x in stages) + common + 1.5 * added_checkpoint,
                epoch_resume_checkpoint_seconds=added_checkpoint, stages=stages,
                caveats=['Compute rates retain Stage 2E measurements; full production training was not benchmarked.',
                         'Primary estimate assumes input prefetch overlaps compute; serial bound adds both and may double-count part of Stage 2E input cost.',
                         'Checkpoint cost uses the first-stage dry-run write plus checksum verification for all 14 epoch checkpoints; serial bound allows 50% more for larger later-stage Adam state.',
                         'Not a confidence interval; thermal variation, background IO and compile/checkpoint variation remain possible.',
                         'Generation and integrity audit are one-time preparation and excluded from training-only hours.'])


def build(root):
    root = Path(root); cache = root / 'cache'
    meta = f.read(cache / 'cache.json'); verification = f.read(cache / 'verification.json')
    loader = f.read(root / 'loader/result.json'); dry = f.read(root / 'dry_run/result.json')
    if meta['status'] != 'COMPLETE' or any(x['status'] != 'PASS' for x in (verification, loader, dry)):
        raise ValueError('Incomplete/failed Stage 2G evidence')
    if meta['samples'] != 145819 or meta['counts'] != f.COUNTS or loader['samples'] != 145819:
        raise ValueError('Full VAL1 count mismatch')
    if verification['cache_catalog_sha256'] != r.sha256_file(cache / 'cache.json'):
        raise ValueError('Verification/catalog checksum mismatch')
    context = f.read(cache / 'context.json')['context']
    for path, expected in context['protected_sha256'].items():
        r.sha256_file(Path(path), expected)
    metadata_bytes = sum(p.stat().st_size for p in cache.iterdir() if p.is_file() and p.suffix != '.npy')
    preserved = list((cache / 'interrupted').glob('*'))
    preserved_bytes = sum(p.stat().st_size for p in preserved if p.is_file())
    previous = f.read('reports/stage2e_performance.json')['forecast']
    predicted = forecast(previous, loader, dry['checkpoint_write_and_hash_seconds'])
    checks = dict(total_count=meta['samples'] == 145819, split_counts=meta['counts'] == f.COUNTS,
                  exact_payload_size=meta['npy_bytes'] == context['expected_storage']['npy_bytes'],
                  full_integrity=verification['full_integrity'] == 'PASS',
                  source_immutability=verification['source_immutability'] == 'PASS',
                  representative_parity=verification['representative']['status'] == 'PASS',
                  random_1000_audit=verification['random_audit']['status'] == 'PASS' and verification['random_audit']['samples'] == 1000,
                  loader_all_samples=loader['status'] == 'PASS',
                  exactly_one_optimizer_update=dry['optimizer_updates_total'] == 1,
                  dry_run=dry['status'] == 'PASS', checkpoint=dry['checkpoint_restore_exact'],
                  portable_export=dry['portable_checkpoint']['status'] == 'PASS',
                  resume=dry['resume_cursor_exact'], protected_files_unchanged=True)
    if not all(checks.values()):
        raise ValueError('A Stage 2G gate failed')
    return dict(timestamp=r.utc_now(), status='PASS', full_cache='PASS', trainer_cache_integration='PASS',
                production_recommendation='GO; recommendation only, production not started',
                cache_output=str(cache.resolve()), samples=meta['samples'], counts=meta['counts'],
                shard_count=len(meta['shards']), npy_bytes=meta['npy_bytes'], metadata_bytes=metadata_bytes,
                active_cache_bytes=meta['npy_bytes'] + metadata_bytes,
                preserved_partial_files=len(preserved), preserved_partial_bytes=preserved_bytes,
                total_cache_bytes=meta['npy_bytes'] + metadata_bytes + preserved_bytes,
                average_payload_bytes_per_sample=meta['npy_bytes'] / meta['samples'],
                generation_wall_seconds=meta['generation_active_wall_seconds'],
                generation_images_per_second=meta['generation_images_per_second'],
                generation_started_at_utc=meta['created_at_utc'], generation_completed_at_utc=meta['completed_at_utc'],
                generation_images_per_second_is_upper_bound=meta.get('generation_timing_is_lower_bound', False),
                interrupted_sessions=meta.get('interrupted_sessions', 0),
                generation_timing_is_lower_bound=meta.get('generation_timing_is_lower_bound', False),
                generation_timing_scope='Sum of perf_counter worker-pool session wall times; excludes preflight, committed-shard resume validation and final catalog/verification scans; downtime excluded',
                generation_workers=4, failures=meta['failures'], retries=meta['retries'],
                cpu_percent_one_core=meta['cpu_percent_one_core'],
                payload_write_bytes_per_second=meta['average_payload_write_bytes_per_second'],
                disk_metric_scope=meta['disk_throughput_scope'],
                context=context, preflight=f.read(cache / 'preflight.json'),
                verification=verification, loader=loader, dry_run=dry, gates=checks,
                estimate_comparison=dict(stage2e_cache_npy_bytes=context['expected_storage']['npy_bytes'],
                                         stage2e_metadata_estimate_bytes=167_000_000,
                                         stage2e_generation_hours=8.87, actual_generation_hours=meta['generation_active_wall_seconds'] / 3600,
                                         stage2e_loader_images_per_second=770.6, actual_loader_images_per_second=loader['images_per_second'],
                                         stage2e_training_hours=previous['epochs_14_seconds'] / 3600,
                                         revised_training_hours=predicted['epochs_14_seconds'] / 3600,
                                         comparison_scope='Stage 2E generation estimate was serial; Stage 2G uses four identical isolated workers. Full-loader measurement includes per-access integrity checks and full-dataset IO.'),
                production_forecast=predicted, VAL2_accessed=False, source_images_modified=False,
                production_training_started=False, strict_evaluator_modified=False,
                stage2f_strict_parity='FAIL unchanged; numerical GPU GO remains a separate material-equivalence judgment',
                operational_changes=['Optional cache-backed production path; preprocessing, model and scientific schedule unchanged.',
                                     'Epoch-boundary model/Adam checkpoints added for safe production resume; no model selection or early stopping.',
                                     'Dry-run checkpoint purpose is disposable and rejected by production resume.',
                                     'All source reads are exact VAL1 paths from authenticated manifests; no VAL2 directory scan.'],
                artifacts=dict(cache_catalog_sha256=r.sha256_file(cache / 'cache.json'),
                               verification_sha256=r.sha256_file(cache / 'verification.json'),
                               loader_sha256=r.sha256_file(root / 'loader/result.json'),
                               dry_run_sha256=r.sha256_file(root / 'dry_run/result.json')))


def render(s):
    e = s['estimate_comparison']; v = s['verification']; dry = s['dry_run']
    lines = ['# Stage 2G full VAL1 cache', '', f"Generated {s['timestamp']}", '',
             f"Full cache: **{s['full_cache']}**. Trainer/cache integration: **{s['trainer_cache_integration']}**. Production: **{s['production_recommendation']}**.", '',
             f"Cache: `{s['cache_output']}`", '',
             '| Item | Verified value |', '|---|---:|',
             f"| Total samples | {s['samples']:,} |", f"| Training | {s['counts']['train']:,} |",
             f"| Internal validation | {s['counts']['internal_validation']:,} |",
             f"| NPY shards | {s['shard_count']} |", f"| NPY bytes | {s['npy_bytes']:,} |",
             f"| Metadata bytes | {s['metadata_bytes']:,} |", f"| Total cache GB (decimal) | {s['total_cache_bytes'] / 1e9:.6f} |",
             f"| Active cache GB (excluding preserved partials) | {s['active_cache_bytes'] / 1e9:.6f} |",
             f"| Preserved interrupted files / bytes | {s['preserved_partial_files']} / {s['preserved_partial_bytes']:,} |",
             f"| Generation hours | {s['generation_wall_seconds'] / 3600:.4f} |",
             f"| Generation images/s | {s['generation_images_per_second']:.3f} |",
             f"| Failures / automatic retries | {s['failures']} / {s['retries']} |", '',
             '## Provenance and integrity', '',
             'Source manifest, both frozen splits, stain reference, preprocessing fingerprint and dependency pins were verified before generation. Full metadata is in the JSON report and cache context. Items retain sample ID, label, split, original relative path and JPEG SHA-256, tensor SHA-256, shard/index, uint8 shape and preprocessing status.', '',
             'Each cache tensor is 350×350×3 uint8 after the unchanged PIL RGB → Pillow LANCZOS → legacy brightness → StainTools Macenko path. Float32 conversion, /255 and stateless flips occur after loading. Validation is never augmented.', '',
             f"All {s['samples']:,} tensor checksums and original VAL1 source JPEG checksums were reverified. Source size/mtime were also checked against generation. Full mapping, counts, shape/dtype, orphan/incomplete-shard and uniqueness checks passed.", '',
             f"Representative audit: **PASS**, {v['representative']['samples']:,} items covering both splits/classes, native dimension strata and every shard. Deterministic random audit: **PASS**, {v['random_audit']['samples']:,} items. Both uint8 tensors and the strict online float32 preprocessing output matched exactly: maximum difference 0; differing elements 0.", '',
             'Committed shards are checksum-verified and retained on resume. Uncommitted shard remnants are preserved in `interrupted/` before rebuilding. The current implementation assumes one cache-generation writer at a time.', '',
             '## Trainer integration and bounded dry run', '',
             f"The actual trainer input path traversed all {s['samples']:,} items with exact order/label/split checks. Physical batch 4, effective batch 100, float32, TF32 disabled, frozen BN, Adam settings and the 14-epoch schedule remain unchanged.", '',
             f"Dry run: 100 training samples, exactly one optimizer update, and {dry['validation_samples']} internal-validation samples. Forward/backward/accumulation and outputs were finite. Frozen weights stayed exact. Model and Adam state were written, deliberately perturbed, restored and hash-checked; predictions matched exactly after restoration. The resume cursor returned the next four expected samples without another optimizer update.", '',
             'A disposable full NASNetLarge `.keras` export was also written and reloaded. All weight hashes and the probe prediction matched exactly; the diagnostic artifact is not a production checkpoint.', '',
             'The cache path is optional in `modern_pca/train_reimplementation.py`; future production requires explicit `--train-production --cache ... --batch-size 4`. The separate `modern_pca.cached_training` CLI performs only the disposable one-update dry run. Production rejects its diagnostic checkpoint.', '',
             '## Performance and forecast', '',
             '| Measurement | Stage 2E estimate/measurement | Stage 2G actual/revised |', '|---|---:|---:|',
             f"| NPY GB | {e['stage2e_cache_npy_bytes'] / 1e9:.6f} | {s['npy_bytes'] / 1e9:.6f} |",
             f"| Metadata MB | 167 (approximate) | {s['metadata_bytes'] / 1e6:.3f} |",
             f"| Generation hours | 8.87 (serial estimate) | {e['actual_generation_hours']:.4f} (4 workers) |",
             f"| Loader images/s | 770.6 (bounded subset) | {e['actual_loader_images_per_second']:.3f} (full cache) |",
             f"| Training-only hours | {e['stage2e_training_hours']:.3f} | {e['revised_training_hours']:.3f} (forecast) |", '',
             s['generation_timing_scope'] + '.', '',
             f"Interrupted sessions: {s['interrupted_sessions']}. Timing is a lower bound: {s['generation_timing_is_lower_bound']}. An abruptly interrupted session retains its last persisted monotonic measurement, excluding its unrecorded tail; in that case the reported generation throughput is correspondingly an upper bound, not an exact timing claim.", '',
             f"Worker CPU utilization summed across workers: {s['cpu_percent_one_core']:.2f}% of one core. Payload write rate averaged over generation: {s['payload_write_bytes_per_second'] / 1e6:.3f} MB/s; this includes compute time and is not a block-device benchmark.", '',
             f"Full-loader timing: {s['loader']['timing_scope']}. Cache temperature: {s['loader']['cache_temperature']}. Reader startup took {s['loader']['reader_startup_seconds']:.3f} seconds.", '',
             f"The forecast includes 14 epoch-resume checkpoints in addition to the final portable model. The conservative serial-input bound is {s['production_forecast']['conservative_serial_seconds'] / 3600:.3f} hours.", '']
    lines.extend('- ' + x for x in s['production_forecast']['caveats'])
    lines.extend(['', '## Scope and checks', '',
                  'VAL2 accessed: **NO**. Source images modified: **NO**. Strict evaluator modified: **NO**. Frozen split changed: **NO**. Production training started: **NO**. Commit/push: **NO**.', '',
                  'Stage 2F strict CPU/GPU parity remains **FAIL** under its original thresholds; its separate numerical GO assessment is unchanged.', '',
                  'Tests, source hashes, `git diff --check`, new-file whitespace checks and final Git status are recorded in `reports/stage2g_checks.json`.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/stage2g_full_cache'))
    args = parser.parse_args(); summary = build(args.run)
    f.atomic_json(Path('reports/stage2g_full_cache.json'), summary)
    Path('reports/stage2g_full_cache.md').write_text(render(summary), encoding='utf-8', newline='\n')
    print('Stage 2G PASS; cache and trainer integration verified; production GO recommendation only.')


if __name__ == '__main__':
    main()
