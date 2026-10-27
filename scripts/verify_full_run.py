"""Verify actual paired generation, three encoders, metrics and replay evidence."""
import argparse
from pathlib import Path

from allbench.features import load_bundle
from allbench.io import atomic_json, file_hash, read_json, require
from allbench.protocol import KINDS, validate

parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--evidence', required=True)
args = parser.parse_args()
root, evidence = Path(args.run), Path(args.evidence)
spec, identity = validate(root / 'generated/benchmark.json')
require(spec.get('evaluation_mode', 'paired') == 'paired', 'PAIRED_REQUIRED')
bundle, arrays = load_bundle(root / 'features/features.json', spec, identity)
require(set(bundle['encoders']) == {'dino', 'clip', 'style'}, 'THREE_REAL_ENCODERS_REQUIRED')
require(not any(e['synthetic'] for e in bundle['encoders'].values()), 'SYNTHETIC_FEATURES_NOT_ACCEPTED')
generated = [r for r in spec['images'] if r['role'] != 'reference']
for row in generated:
    receipt = read_json(root / 'generated' / row['generation_receipt'])
    require(receipt['runtime']['adapter_load']['weights_verified'], 'LORA_MOUNT_NOT_VERIFIED')
    require(receipt['metrics']['adapter_enabled'] == (row['role'] == 'adapted'), 'ADAPTER_SWITCH_MISMATCH')
pairs = {}
for row in generated:
    pairs.setdefault((row['prompt_id'], row['seed']), {})[row['role']] = row['sha256']
changed = sum(p['base'] != p['adapted'] for p in pairs.values())
require(changed > 0, 'ALL_PAIRED_OUTPUTS_IDENTICAL')
report = read_json(root / 'evaluation/report.json')
for row in report['groups']:
    for metric in ('fidelity', 'style_fidelity'):
        require(all(row[metric][split]['status'] == 'measured' for split in ('train', 'holdout')), 'MISSING_FIDELITY')
    require(row['text_alignment']['status'] == row['diversity']['status'] == 'measured', 'MISSING_METRIC')
    if row['role'] == 'adapted':
        require(row['base_style_drift']['status'] == 'measured', 'MISSING_STYLE_DRIFT')
require(all(set(k for k, n in counts.items() if n) == KINDS for counts in report['prompt_coverage'].values()), 'SIX_PROMPT_KINDS_REQUIRED')
for name in ('generation-partial', 'encoding-partial'):
    proof = read_json(evidence / (name + '.json'))
    require(proof['files'], 'EMPTY_RECOVERY_PROOF')
    for path, sha in proof['files'].items():
        require(file_hash(root / path) == sha, 'RECOVERED_FILE_CHANGED')
replay = read_json(evidence / 'replay.json')['stages']
require(replay['generate']['generated_this_attempt'] == 0 and replay['generate']['load_seconds_this_attempt'] == 0, 'REPLAY_GENERATED')
require(replay['encode']['encoded_images'] == replay['encode']['encoded_texts'] == 0, 'REPLAY_ENCODED')
require(not replay['evaluate']['computed'] and replay['report']['reused'], 'REPLAY_RECOMPUTED')
checkpoint = read_json(root / 'evaluation/checkpoint.json')
require(checkpoint['report'] == report, 'EVALUATION_CHECKPOINT_CHANGED')
result = {'status': 'passed', 'generated': len(generated), 'references': len(spec['images']) - len(generated),
          'changed_pairs': changed, 'pairs': len(pairs), 'encoded_images': len(arrays['images']),
          'encoded_texts': len(arrays['texts']), 'report_groups': len(report['groups']),
          'real_encoders': sorted(bundle['encoders']), 'generation_identity': spec['generator_provenance']['identity'],
          'adapter_load': spec['generator_provenance']['runtime']['adapter_load'],
          'report_sha256': file_hash(root / 'evaluation/report.json'),
          'all_recovery_and_zero_replay_checks': True, 'model_effect_validated': False,
          'synthetic_reference_fixture': spec['synthetic']}
atomic_json(evidence / 'verification-summary.json', result)
print(result)
