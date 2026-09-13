"""Fail-closed material readiness checks; never starts a trainer or GPU.

This verifies source/review bindings and release interfaces. Semantic judgments
remain independently authored evidence, not a consequence of passing Python.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.validate_rl_curation import canonical_hash,load_rows,as_output
from tools.assemble_rl_release import OUT,approved,read_json,write_json,case_partition,annotation_errors
from rl_materials.contracts import validate_output,resolve_annotations

FROZEN_SPLITS={
    'train':'69c780015eb37da06257cb33795963d77db5d8b60fd0602620ab18b792b561d3',
    'validation':'69c28ae4b5205a9a13ed5cec2081231936e991a123100f8264c6b54e110d2db0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_errors(directory,manifest):
    errors=[]
    files=manifest.get('files',{})
    if not files or manifest.get('content_digest')!=canonical_hash(files):errors.append('manifest_digest')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p!=directory/'manifest.json'}
    for name in sorted(actual-set(files)):errors.append('manifest_unlisted_member:'+name)
    for name,digest in files.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()):
            errors.append('manifest_path_escape:'+name)
        elif not path.is_file() or sha(path)!=digest:
            errors.append('manifest_member:'+name)
    return errors


def policy_errors(policy,reference):
    errors=[]
    if set(policy)!={'episode_id','task_type','images','messages'}:errors.append('policy_keys')
    messages=policy.get('messages',[])
    if not messages or any(set(m)!={'role','content'} or m['role']!='user' or not isinstance(m['content'],str) for m in messages):
        errors.append('policy_message_schema_or_assistant_leak')
    text=json.dumps(messages,ensure_ascii=False)
    answer=reference.get('response',{})
    if policy['task_type']=='patent_advisory':
        if answer.get('answer') and answer['answer'] in text:errors.append('reference_answer_leak')
        if policy['images']:errors.append('advisory_images')
    else:
        for claim in answer.get('claims',[]):
            if claim['text'].strip() and claim['text'] in text:errors.append('reference_claim_leak')
    return errors


def calibration_errors(directory,current_pairs,current_prompts=None,current_refs=None):
    errors=[]
    prep=read_json(directory/'preparation.json',{})
    report=read_json(directory/'report.json',{})
    labels=read_json(directory/'labels.PRIVATE.json',[])
    if not report.get('passed') or report.get('comparisons',0)<20 or report.get('agreement',0)<.85:
        errors.append('blind_calibration_not_passed')
    if not report.get('judge_configuration'):errors.append('judge_configuration_missing')
    if len(labels)!=len(current_pairs):errors.append('calibration_pair_count_stale')
    pairs={r['pair_id']:canonical_hash(r) for r in current_pairs}
    if {r['source_pair_id']:r['pair_sha256'] for r in labels}!=pairs:
        errors.append('calibration_pairs_stale')
    if current_prompts is not None:
        prompts={p['episode_id']:p for p in current_prompts}
        blind_path=directory/'blind.jsonl'
        blind={r['comparison_id']:r for r in load_rows(blind_path)} if blind_path.is_file() else {}
        current={r['pair_id']:r for r in current_pairs}
        for label in labels:
            pair=current.get(label['source_pair_id']);p=prompts.get(label['source_episode_id'])
            if not pair or not p:continue
            ref=current_refs[p['episode_id']];side=label['expected']
            expected={'comparison_id':label['comparison_id'],'task_type':p['task_type'],
                'messages':p['messages'],'images':[str((OUT/x).resolve()) for x in p['images']],
                'rubric':{'required_points':ref['required_points'],'forbidden_claims':ref['forbidden_claims']},
                'candidate_A':pair['chosen'] if side=='A' else pair['rejected'],
                'candidate_B':pair['rejected'] if side=='A' else pair['chosen']}
            expected['input_sha256']=canonical_hash(expected)
            if blind.get(label['comparison_id'])!=expected:errors.append('calibration_policy_or_rubric_stale:'+label['comparison_id'])
    for filename,key in [('blind.jsonl','blind_sha256'),('labels.PRIVATE.json','labels_sha256'),('decisions.json','decisions_sha256')]:
        if not (directory/filename).is_file() or sha(directory/filename)!=report.get(key):errors.append('calibration_integrity:'+filename)
    if sha(ROOT/'rl_materials/REWARD_PROTOCOL.md')!=report.get('protocol_sha256') or report.get('protocol_sha256')!=prep.get('protocol_sha256'):
        errors.append('calibration_protocol_stale')
    return errors


def audit(round_name='round1',output=None):
    errors=[];checks={};dependencies={}
    def check(name,issues):
        checks[name]={'passed':not issues,'errors':issues}
        errors.extend(name+':'+x for x in issues)
    def bind(path):
        p=Path(path);dependencies[str(p.resolve())]=sha(p)
    manifest=read_json(OUT/'manifest.json',{})
    check('manifest',manifest_errors(OUT,manifest))
    bind(OUT/'manifest.json')
    lanes={'kr':'kr_cases.jsonl','us':'us_cases.jsonl','drawings':'claimsets.jsonl','new_test':'claimsets_test.jsonl','annotation':'annotation_extensions.jsonl'}
    source_rows={};lane_counts={};issues=[]
    for lane,name in lanes.items():
        path=ROOT/'rl_materials/curation'/name
        rows=load_rows(path) if path.is_file() else []
        lane_counts[lane]=len(rows)
        if path.is_file():bind(path)
        for suffix in ('semantic-review.json','source-schema-audit.json'):
            p=ROOT/f'.superloopy/evidence/{lane}-{suffix}'
            if p.is_file():bind(p)
        for row in rows:
            rid=row.get('case_id',row.get('record_id'))
            if rid in source_rows:issues.append('duplicate_source_id:'+rid)
            source_rows[rid]=row
            ok,reason=approved(row,lane)
            if not ok:issues.append(lane+':'+rid+':'+str(reason))
            if lane in {'kr','us'}:
                pdf=Path(row['pdf_path'])
                if not pdf.is_file() or sha(pdf)!=row['source_sha256']:issues.append('case_pdf_changed:'+rid)
                else:bind(pdf)
    minimum={'kr':20,'us':20,'drawings':40,'new_test':8,'annotation':16}
    issues += ['coverage:'+k for k,n in minimum.items() if lane_counts.get(k,0)<n]
    check('independent_source_review',issues)
    from tools.build_rl_lineage import build
    lineage=build()
    issues=[]
    if not lineage.get('passed') or lineage.get('errors'):issues.append('lineage_audit_failed')
    if lineage.get('protected_patents')!=694 or lineage.get('old_test_patent_count')!=75:issues.append('protected_inventory')
    bind(ROOT/'data/case_rl/lineage_audit.json')
    for item in lineage.get('metadata_sources',[]):
        p=ROOT/item['path']
        if not p.is_file() or sha(p)!=item['sha256']:issues.append('metadata_changed:'+item['path'])
        else:bind(p)
    overrides=ROOT/'rl_materials/curation/family_overrides.json'
    family_review=read_json(ROOT/'.superloopy/evidence/family-overrides-review.json',{})
    bind(overrides);bind(ROOT/'.superloopy/evidence/family-overrides-review.json')
    if (family_review.get('independent_from_author') is not True or not family_review.get('reviewer')
        or family_review.get('reviewer')==family_review.get('author_agent')
        or family_review.get('source_overrides_sha256')!=sha(overrides)
        or lineage.get('overrides_sha256')!=sha(overrides)
        or len(family_review.get('records',[]))!=6
        or any(r.get('verdict')!='accept' for r in family_review.get('records',[]))):
        issues.append('official_overrides_review')
    for r in read_json(overrides)['records']:
        p=ROOT/r['source_path']
        if not p.is_file() or sha(p)!=r['source_sha256']:issues.append('official_override_source:'+r['patent_id'])
        else:bind(p)
    for split,digest in FROZEN_SPLITS.items():
        p=ROOT/f'data/rl_source_v2/hf_multimodal/{split}.jsonl'
        if not p.is_file() or sha(p)!=digest:issues.append('frozen_source_changed:'+split)
        else:bind(p)
    bind(ROOT/'data/case_rl/families/protected_ids.json')
    image_integrity=ROOT/'data/rl_source_v2/image_integrity.json'
    frozen_images=read_json(image_integrity,{})
    bind(image_integrity);bind(ROOT/'data/rl_source_v2/download_revision.json')
    downloaded=read_json(ROOT/'data/rl_source_v2/download_revision.json',{})
    if (frozen_images.get('test_opened') is not False or frozen_images.get('scope')!='train_validation'
        or downloaded.get('revision')!='e978f526d0b15f6c981a4b82fd25404cef68a8d7'):
        issues.append('frozen_image_provenance')
    expected_images={r['path']:r['sha256'] for r in frozen_images.get('images',[])}
    for r in source_rows.values():
        if r.get('source_split') in {'train','validation'} and 'images' in r:
            for rel in r['images']:
                p=ROOT/'data/rl_source_v2'/rel
                if not p.is_file() or sha(p)!=expected_images.get(rel):issues.append('frozen_image_changed:'+rel)
    new_provenance=ROOT/'data/case_rl/new_claim_test/provenance.jsonl'
    bind(new_provenance);bind(ROOT/'data/case_rl/new_claim_test/manifest.jsonl')
    for r in load_rows(new_provenance):
        source=source_rows.get(r['record_id'],{})
        if source.get('images')!=[p['path'] for p in r['images']]:issues.append('new_test_source_order:'+r['record_id'])
        pid=r['record_id'].removeprefix('NEW-').removesuffix('-claimset')
        if sha(ROOT/f'data/case_rl/families/{pid}.json')!=r['family_metadata_sha256']:issues.append('new_test_metadata_changed:'+pid)
        for image in r['images']:
            p=ROOT/image['path']
            if not p.is_file() or sha(p)!=image['sha256']:issues.append('new_test_image_changed:'+image['path'])
    check('family_and_frozen_boundaries',issues)
    policies={};refs={};pairs={};cards={};splits={};issues=[]
    registry=read_json(OUT/'audit/registry.json',[])
    reg={r['episode_id']:r for r in registry}
    quarantine=read_json(OUT/'audit/quarantined.json',[])
    if quarantine:issues.append('quarantined_records:'+str(len(quarantine)))
    seen=set();family_splits=defaultdict(set);counts={};annotation_counts=Counter();case_counts=defaultdict(Counter)
    all_lineage={r.get('case_id',r.get('record_id')):r for key in ('case_lineages','selected_drawings','new_test') for r in lineage.get(key,[])}
    for split in ('train','calibration','test'):
        policies[split]=load_rows(OUT/f'policy/{split}.jsonl')
        ref_rows=load_rows(OUT/f'reward/references.{split}.jsonl');refs[split]={r['episode_id']:r for r in ref_rows}
        pairs[split]=load_rows(OUT/f'reward/preferences.{split}.jsonl')
        card_rows=load_rows(OUT/f'evidence/cards.{split}.jsonl');cards[split]={r['card_id']:r for r in card_rows}
        if len(refs[split])!=len(ref_rows) or len(cards[split])!=len(card_rows):issues.append('duplicate_reward_or_card_id:'+split)
        counts[split]={'episodes':len(policies[split]),'preferences':len(pairs[split]),
            'task_types':dict(Counter(r['task_type'] for r in policies[split]))}
        if {r['episode_id'] for r in policies[split]}!=set(refs[split]):issues.append('policy_reference_alignment:'+split)
        for p in policies[split]:
            eid=p['episode_id'];ref=refs[split].get(eid,{})
            if eid in seen:issues.append('episode_cross_split_duplicate:'+eid)
            seen.add(eid);splits[eid]=split
            issues += [eid+':'+e for e in policy_errors(p,ref)]
            info=reg.get(eid,{})
            source=source_rows.get(info.get('record_id'))
            if not source or info.get('source_record_sha256')!=canonical_hash(source) or info.get('split')!=split:
                issues.append('registry_binding:'+eid);continue
            li=all_lineage.get(info['record_id'],{})
            if info.get('lineage_key')!=li.get('lineage_key') or not li.get('lineage_key'):issues.append('lineage_registry_binding:'+eid)
            if li.get('record_sha256') and li['record_sha256']!=canonical_hash(source):issues.append('lineage_source_stale:'+eid)
            if any(li.get(k) for k in ('missing_metadata','old_test_overlap','old_test_family_overlap','validation_to_train_overlap','case_overlap')):issues.append('lineage_overlap:'+eid)
            family_splits[info['lineage_key']].add(split)
            if p['task_type']=='patent_advisory':
                ep=next((e for e in source['advisory_episodes'] if e['episode_id']==eid),None)
                if not ep or ref['response']!={'answer':ep['reference_answer'],'citations':ep['evidence_ids'],'limitations':[ep['temporal_scope']]}:issues.append('advisory_reference_changed:'+eid)
                text='\n'.join(m['content'] for m in p['messages'])
                if ep and (ep['question'] not in text or ep['temporal_scope'] not in text):issues.append('advisory_scenario_missing:'+eid)
                case_counts[info['jurisdiction']][split]+=1
            else:
                expected=as_output(source['claims'],source['parents'],source['abstentions'])
                context=json.loads(p['messages'][-1]['content'])
                supplied=context['case_cards']
                supplied_map={c['card_id']:{**c,'review_status':'approved'} for c in supplied}
                if len(supplied_map)!=len(supplied):issues.append('duplicate_supplied_card:'+eid)
                ext=source_rows.get(info.get('annotation_record_id'))
                if ext:
                    annotation_counts[split]+=1
                    expected['annotations']=ext['annotations']
                    if canonical_hash(ext)!=info.get('annotation_record_sha256'):issues.append('annotation_registry_stale:'+eid)
                    issues += [eid+':'+e for e in annotation_errors(ext,source,cards['train'],split,li['lineage_key'])]
                    if set(supplied_map)!=set(ext['card_ids']):issues.append('supplied_annotation_cards:'+eid)
                elif supplied:issues.append('unreviewed_annotation_context:'+eid)
                for c in supplied:
                    train_card=cards['train'].get(c['card_id'])
                    if not train_card or any(c.get(k)!=train_card.get(k) for k in c):issues.append('supplied_card_stale_or_heldout:'+eid)
                if ref['response']!=expected:issues.append('drawing_reference_changed:'+eid)
                issues += [eid+':'+e for e in validate_output(ref['response'],supplied_map)]
                if expected['annotations']:
                    try:resolve_annotations(expected,supplied_map)
                    except (ValueError,KeyError):issues.append('annotation_source_resolver:'+eid)
                if len(p['images'])!=len(source['images']):issues.append('image_count:'+eid)
                for original,rel in zip(source['images'],p['images']):
                    image=ROOT/original if source['source_split']=='new_test' else ROOT/'data/rl_source_v2'/original
                    if not image.is_file() or not (OUT/rel).is_file() or sha(image)!=sha(OUT/rel):issues.append('image_order_or_integrity:'+eid)
        pair_ids=set()
        for pair in pairs[split]:
            if pair['pair_id'] in pair_ids:issues.append('duplicate_pair:'+pair['pair_id'])
            pair_ids.add(pair['pair_id'])
            ref=refs[split].get(pair['episode_id'])
            if not ref or pair['chosen']!=ref['response'] or pair['chosen']==pair['rejected'] or not pair.get('preference_reason'):
                issues.append('invalid_preference:'+pair['pair_id'])
    if len(reg)!=len(registry) or set(reg)!=seen:issues.append('registry_episode_set')
    if any(len(v)>1 for v in family_splits.values()):issues.append('family_cross_split')
    expected_tasks={'train':{'claim_set':32,'patent_advisory':56},'calibration':{'claim_set':8,'patent_advisory':12},'test':{'claim_set':8,'patent_advisory':12}}
    for split,expected in expected_tasks.items():
        if counts[split]['task_types']!=expected:issues.append('task_coverage:'+split)
    if counts['train']['preferences']<100 or counts['calibration']['preferences']<20:issues.append('preference_coverage')
    for country in ('KR','US'):
        if dict(case_counts[country])!={'train':28,'calibration':6,'test':6}:issues.append('case_split_coverage:'+country)
    for split,n in {'train':8,'calibration':4,'test':4}.items():
        if annotation_counts[split]<n:issues.append('annotation_coverage:'+split)
    check('release_content_and_input_isolation',issues)
    probe=read_json(ROOT/'.superloopy/evidence/release-processor-probe.json',{})
    issues=[]
    if not probe.get('passed') or probe.get('sample_only') or probe.get('device')!='cpu' or probe.get('model_weights_loaded') is not False:
        issues.append('full_cpu_probe_missing')
    probe_rows={r['episode_id']:r for r in probe.get('records',[])}
    if len(probe_rows)!=len(seen) or set(probe_rows)!=seen:issues.append('processor_probe_coverage')
    for split in policies:
        for p in policies[split]:
            r=probe_rows.get(p['episode_id'],{})
            if r.get('policy_sha256')!=canonical_hash(p) or r.get('reward_reference_sha256')!=canonical_hash(refs[split][p['episode_id']]):issues.append('processor_probe_stale:'+p['episode_id'])
    probe_path=ROOT/'.superloopy/evidence/release-processor-probe.json'
    if probe_path.is_file():bind(probe_path)
    check('actual_processor_encoding',issues)
    directory=ROOT/'.superloopy/evidence/judge_calibration'/round_name
    # Recompute scores from the preserved decisions and labels; do not trust a
    # handwritten or stale report's passed boolean.
    if all((directory/x).is_file() for x in ('preparation.json','blind.jsonl','labels.PRIVATE.json','decisions.json')):
        from tools.calibrate_rl_material_judge import score
        score(round_name)
    check('blind_reward_calibration',calibration_errors(directory,pairs['calibration'],policies['calibration'],refs['calibration']))
    for name in ('preparation.json','blind.jsonl','labels.PRIVATE.json','decisions.json','report.json'):
        p=directory/name
        if p.is_file():bind(p)
    for p in [ROOT/'rl_materials/MATERIALS_SPEC.md',ROOT/'rl_materials/REWARD_PROTOCOL.md',*sorted((ROOT/'tools').glob('*rl*release*.py')),
              ROOT/'tools/calibrate_rl_material_judge.py',ROOT/'tools/validate_rl_curation.py',ROOT/'tools/validate_rl_annotations.py',ROOT/'tools/build_rl_lineage.py']:
        bind(p)
    report={'materials_complete':not errors,'gpu_allowed':False,'model_quality':'unmeasured',
        'scope':'Reviewed pilot RL reward materials, not full-corpus semantic review or a trained/evaluated model.',
        'checks':checks,'errors':errors,'counts':counts,'source_counts':lane_counts,
        'annotation_episodes':dict(annotation_counts),
        'claim_counts':{split:dict(Counter(c['kind'] for r in refs[split].values() if r['task_type']=='claim_set' for c in r['response']['claims'])) for split in refs},
        'content_digest':manifest.get('content_digest'),
        'dependencies':dependencies,'calibration_round':round_name,
        'starting_adapter':'Mepeng22/gemma-4-31b-claim-lora-v2@cc04579366bbb88663029d53b2aafcc84159589e'}
    write_json(output or ROOT/'.superloopy/evidence/material-release-audit.json',report)
    print(json.dumps({**{k:v for k,v in report.items() if k not in {'dependencies','checks','errors'}},
        'error_count':len(errors),'failed_checks':[k for k,v in checks.items() if not v['passed']]},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--round',default='round1')
    args=parser.parse_args()
    raise SystemExit(0 if audit(args.round)['materials_complete'] else 1)
