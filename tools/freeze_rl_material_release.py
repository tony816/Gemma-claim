"""Freeze an audited, reviewed RL material package without GPU/model actions."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.assemble_rl_release import OUT,write_json,read_json
from tools.validate_rl_curation import canonical_hash
from tools.audit_rl_material_release import audit,sha,manifest_errors


def review_gate_errors(report):
    errors=[]
    review=read_json(ROOT/'.superloopy/evidence/release-code-review.json',{})
    files=review.get('reviewed_files',{})
    required=['tools/'+name for name in ('assemble_rl_release.py','audit_rl_material_release.py','freeze_rl_material_release.py',
              'validate_rl_annotations.py','build_rl_lineage.py','rl_family_metadata.py',
              'calibrate_rl_material_judge.py','probe_rl_material_release.py','validate_rl_curation.py')]
    required += ['tests/test_rl_material_release.py','tests/test_rl_family_metadata.py',
                 'rl_materials/MATERIALS_SPEC.md','rl_materials/REWARD_PROTOCOL.md']
    if (review.get('recommendation')!='APPROVE' or review.get('codeQualityStatus')!='CLEAR'
        or not review.get('reviewer') or review.get('reviewer')=='/root'
        or review.get('independent_from_author') is not True):errors.append('independent_code_review_missing')
    for rel in required:
        if files.get(rel)!=sha(ROOT/rel):errors.append('code_review_stale:'+rel)
    qa=read_json(ROOT/'.superloopy/evidence/release-qa.json',{})
    if (qa.get('passed') is not True or not qa.get('reviewer') or qa.get('reviewer')=='/root'
        or qa.get('independent_from_author') is not True
        or qa.get('candidate_content_digest')!=report['content_digest']):errors.append('independent_release_qa_missing_or_stale')
    scenarios=qa.get('scenarios',[])
    if len(scenarios)<3 or any(r.get('passed') is not True or not r.get('evidence') for r in scenarios):errors.append('release_qa_scenarios_missing')
    return errors


def freeze(round_name):
    report=audit(round_name)
    if not report['materials_complete']:
        raise ValueError('Materials audit failed; nothing frozen.')
    errors=review_gate_errors(report)
    if errors:raise ValueError('Independent release reviews failed: '+str(errors))
    report['release_reviews']={name:sha(ROOT/'.superloopy/evidence'/name)
        for name in ('release-code-review.json','release-qa.json')}
    directory=ROOT/'data/case_rl/releases'/('rl-pilot-'+canonical_hash(report)[:16])
    if directory.exists():
        manifest=read_json(directory/'manifest.json',{})
        errors=manifest_errors(directory,manifest)
        if errors:raise ValueError('Existing frozen release differs: '+str(errors))
        print(json.dumps({'release':str(directory),'existing_immutable_release':True}))
        return directory
    directory.mkdir(parents=True)
    source_manifest=read_json(OUT/'manifest.json')
    for name in source_manifest['files']:
        target=directory/('audit/assembly_snapshot.json' if name=='ASSEMBLY_STATUS.json' else name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OUT/name,target)
    # Portable evidence for reviewers; the policy loader must use policy/* only.
    for p in sorted((ROOT/'rl_materials/curation').glob('*.json*')):
        target=directory/'audit/curation'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in sorted((ROOT/'.superloopy/evidence').glob('*-semantic-review.json')):
        target=directory/'audit/reviews'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in sorted((ROOT/'.superloopy/evidence').glob('*-source-schema-audit.json')):
        target=directory/'audit/reviews'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for p in [ROOT/'data/case_rl/lineage_audit.json',ROOT/'.superloopy/evidence/family-overrides-review.json',
              ROOT/'.superloopy/evidence/release-processor-probe.json',
              ROOT/'.superloopy/evidence/release-code-review.json',ROOT/'.superloopy/evidence/release-qa.json']:
        shutil.copyfile(p,directory/'audit'/p.name)
    calibration=ROOT/'.superloopy/evidence/judge_calibration'/round_name
    for name in ('preparation.json','blind.jsonl','labels.PRIVATE.json','decisions.json','report.json'):
        target=directory/'audit/calibration'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(calibration/name,target)
    write_json(directory/'MATERIAL_STATUS.json',{k:v for k,v in report.items() if k!='dependencies'})
    write_json(directory/'audit/build_dependencies.json',report['dependencies'])
    description='''# Gemma v2 reinforcement-learning materials — reviewed pilot

This package contains RL prompts, multimodal inputs, grounded reward references,
and substantive preference comparisons. It is not an SFT or GPU training run.
Continue the existing v2 adapter named in MATERIAL_STATUS.json only in a later,
separately configured training run. GPU authorization remains false here.

- `policy/train.jsonl`: only policy-visible questions, drawings and provided
  source evidence. Resolve image paths against this release directory.
- `reward/references.train.jsonl`: source-grounded reference responses and
  semantic rubrics, for the reward/judge channel only.
- `reward/preferences.train.jsonl`: preferred/inferior responses with substantive
  reasons. Can support preference-based RL alignment or reward supervision.
- `policy/calibration.jsonl` and matching reward files: held-out judge calibration.
- `policy/test.jsonl` and matching reward files: held-out material evaluation;
  never use its answers or preferences to fit the policy or reward model.
- `evidence/cards.*.jsonl`: source metadata and exact page excerpts. Case cards
  supplied to drawing tasks come only from TRAIN, match jurisdiction, and exclude
  the target patent family. Historical law scope is explicit.
- `REWARD_PROTOCOL.md`: semantic dimensions, abstention and severe-error caps.
- `audit/`: provenance and review evidence, private gold labels and build snapshot.
  Never concatenate this directory into policy prompts. Assembly snapshot is the
  pre-audit state; MATERIAL_STATUS.json records the completed material audit.

Case annotations describe supplied case evidence and its application to a claim.
They do not identify what caused an answer inside trained weights. Independent
and dependent reference claims are grounded in the allowed drawings only.

The 40 selected cases and 48 claim-set examples are a reviewed pilot, not semantic
coverage of the entire Drive corpus. The new eight claim tests emphasize laboratory
devices. Model capability, online RL execution and current legal status remain
unmeasured. Blind agreement reports apply to the recorded native judge protocol.

`manifest.json` hashes every package member. Treat this directory as immutable;
changes require a new curation/review/calibration cycle and a new release.
'''
    (directory/'README.md').write_text(description,encoding='utf-8')
    files={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file()}
    manifest={'files':files,'content_digest':canonical_hash(files),'candidate_content_digest':report['content_digest']}
    write_json(directory/'manifest.json',manifest)
    if manifest_errors(directory,manifest):raise ValueError('Frozen copy integrity failed')
    receipt={'release':str(directory),'manifest_sha256':sha(directory/'manifest.json'),
        'content_digest':manifest['content_digest'],'materials_complete':True,'gpu_allowed':False,
        'model_quality':'unmeasured','files':len(files)}
    write_json(ROOT/'.superloopy/evidence/frozen-release.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    return directory


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--round',default='round1')
    freeze(parser.parse_args().round)
