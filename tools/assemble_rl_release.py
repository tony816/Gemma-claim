"""Assemble reviewed RL material candidates, separating policy and reward data.

No GPU/trainer/inference calls. Only exact-hash independently accepted records
with fresh source/schema evidence enter the candidate. The candidate remains
quarantined until the final material audit, blind calibration and processor
probe pass. Rejected drafts remain in the authoring directory, not this export.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.validate_rl_curation import canonical_hash, load_rows, as_output
from rl_materials.contracts import validate_output

OUT = ROOT / 'data/case_rl/release_candidate'


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def approved(row, lane):
    rid = row.get('case_id', row.get('record_id'))
    expected = canonical_hash(row)
    review = read_json(ROOT/f'.superloopy/evidence/{lane}-semantic-review.json', {})
    if review.get('independent_from_author') is not True or not review.get('reviewer'):
        return False, 'independent_review_missing'
    if review['reviewer'] in (review.get('author_agent'), row.get('author')):
        return False, 'reviewer_is_author'
    decisions = [r for r in review.get('records',[]) if r.get('record_id') == rid]
    if len(decisions) != 1 or decisions[0].get('verdict') != 'accept':
        return False, 'semantic_review_not_accepted'
    decision = decisions[0]
    if decision.get('source_record_sha256') != expected or not decision.get('evidence_checked'):
        return False, 'semantic_review_stale_or_no_evidence'
    audit = read_json(ROOT/f'.superloopy/evidence/{lane}-source-schema-audit.json', {})
    checks = [r for r in audit.get('record_hashes',[]) if r.get('record_id') == rid]
    if len(checks) != 1 or checks[0].get('sha256') != expected or checks[0].get('errors'):
        return False, 'source_schema_audit_stale_or_failed'
    return True, {'reviewer':review['reviewer'], 'record_sha256':expected,
                  'review_artifact':f'.superloopy/evidence/{lane}-semantic-review.json'}


def annotation_errors(extension, drawing, supplied, split, lineage_key):
    """References must be supported by current, train-only, unrelated cards."""
    errors=[]
    if not lineage_key:errors.append('annotation_target_lineage_missing')
    if extension.get('drawing_record_id') != drawing['record_id'] or extension.get('drawing_record_sha256') != canonical_hash(drawing):
        errors.append('annotation_drawing_binding')
    if extension.get('source_split') != drawing['source_split']:
        errors.append('annotation_split')
    ids=extension.get('card_ids',[])
    if not ids or len(ids)!=len(set(ids)) or any(cid not in supplied for cid in ids):
        return errors+['annotation_supplied_card_ids']
    selected={cid:supplied[cid] for cid in ids}
    cases={c['case_id']:c['review']['record_sha256'] for c in selected.values()}
    if cases != extension.get('case_record_sha256'):
        errors.append('annotation_case_binding')
    if any(c['jurisdiction']!=extension.get('jurisdiction') for c in selected.values()):
        errors.append('annotation_card_jurisdiction')
    if any(c['lineage_key']==lineage_key for c in selected.values()):
        errors.append('annotation_target_family')
    for field in ('annotations','inferior_annotations'):
        response=as_output(drawing['claims'],drawing['parents'],drawing['abstentions'])
        response['annotations']=extension.get(field)
        errors += [field+':'+x for x in validate_output(response,selected)]
        if not response['annotations']:
            errors.append(field+':empty')
    if extension.get('annotations')==extension.get('inferior_annotations'):
        errors.append('annotation_comparison_identical')
    for field in ('required_points','forbidden_claims','preference_reason'):
        if not extension.get(field):errors.append('annotation_missing:'+field)
    return errors


def asset(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Image missing: '+str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rel = f'assets/{digest}{path.suffix.lower()}'
    destination = OUT/rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(path, destination)
    elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError('Existing asset hash mismatch')
    return rel, {'source_path':path.relative_to(ROOT).as_posix(), 'path':rel,'sha256':digest}


def excerpt(case, span):
    return {'evidence_id':span['id'],'case_id':case['case_id'], 'jurisdiction':case['jurisdiction'],
            'case_name':case['case_name'],'docket':case['docket'],'court':case['court'],
            'decision_date':case['decision_date'],'pdf_page':span['pdf_page'], 'speaker':span['speaker'],
            'quote':span['quote'],'source_sha256':case['source_sha256'], 'source_path':case['pdf_path']}


def case_partition(case, index):
    notes = case.get('lineage_notes')
    named = notes.get('partition') if isinstance(notes,dict) else None
    return named if named in {'train','calibration','test'} else ('train' if index<14 else 'calibration' if index<17 else 'test')


CLAIM_INSTRUCTION = ('제공된 도면을 근거로 독립항1개와 뒷받침되는 종속항을 최대3개 작성하세요. '
    '각 종속항은 앞선 청구항 하나를 인용하고 그 구성을 유지하면서 한정을 추가해야 합니다. '
    '도면 참조부호는 청구항에 쓰지 마세요. 도면에 없는 구조·재질·수치·성능을 만들지 마세요. '
    'JSON으로 claims, annotations, abstentions를 반환하세요. claims의 각 항목에는 '
    'number, kind(independent/dependent), depends_on(독립항은빈배열), text를 넣으세요. '
    'annotations는 제공된 판례 카드만 선택하고 claim_number, card_id, '
    'mode(provided_during_drafting), application을 포함하세요. 카드가 없으면 빈배열입니다. '
    'abstentions에는 확인할 수 없는 사항을 구체적으로 적으세요.')


def assemble():
    policy, references, pairs, cards = (defaultdict(list) for _ in range(4))
    registry, quarantined, image_sources, source_reviews = [], [], {}, []
    lineage = read_json(ROOT/'data/case_rl/lineage_audit.json', {})
    lineage_cases = {r['case_id']:r for r in lineage.get('case_lineages',[])}
    lineage_drawings = {r['record_id']:r for r in lineage.get('selected_drawings',[])}
    lineage_new = {r['record_id']:r for r in lineage.get('new_test',[])}
    all_case_splits = {}
    case_rows = []
    for lane in ('kr','us'):
        rows = load_rows(ROOT/f'rl_materials/curation/{lane}_cases.jsonl')
        for index, case in enumerate(rows):
            split = case_partition(case,index)
            all_case_splits[case['case_id']] = split
            case_rows.append((lane,case,split))
    lineage_splits=defaultdict(set)
    for _,case,split in case_rows:
        info=lineage_cases.get(case['case_id'],{})
        if info.get('record_sha256') == canonical_hash(case):
            lineage_splits[info['lineage_key']].add(split)
    split_conflicts = {k:sorted(v) for k,v in lineage_splits.items() if len(v)>1}
    for lane,case,split in case_rows:
        eligible, proof = approved(case,lane)
        info = lineage_cases.get(case['case_id'],{})
        if not info or info.get('record_sha256') != canonical_hash(case) or info.get('missing_metadata'):
            eligible,proof=False,'case_lineage_missing_or_stale'
        if info.get('lineage_key') in split_conflicts:
            eligible,proof=False,'case_lineage_crosses_splits'
        if info.get('old_test_overlap'):
            eligible,proof=False,'case_overlaps_old_test_family'
        if not eligible:
            quarantined.append({'record_id':case['case_id'],'lane':lane,'reason':proof})
            continue
        source_reviews.append(proof)
        spans = {s['id']:excerpt(case,s) for s in case['evidence_spans']}
        for principle in case['principles']:
            cards[split].append({'card_id':principle['id'],'case_id':case['case_id'],
                'jurisdiction':case['jurisdiction'],'statement':principle['statement'],
                'scope':principle['scope'],'non_rules':principle['non_rules'],
                'history_scope':case['history_scope'], 'evidence':[spans[x] for x in principle['evidence_ids']],
                'review_status':'approved','lineage_key':info['lineage_key'],
                'docket':case['docket'],'court':case['court'],'decision_date':case['decision_date'],
                'pdf_path':case['pdf_path'],'source_sha256':case['source_sha256'],
                'pages':sorted(set(spans[x]['pdf_page'] for x in principle['evidence_ids'])),
                'review':proof,'case_patent_lookup_ids':info['lookup_ids'],
                'usage':'historical_scope_only; target_family_exclusion_required'})
        for ep in case['advisory_episodes']:
            eid=ep['episode_id']
            context=[spans[x] for x in ep['evidence_ids']]
            text=('다음은 판례에 근거한 특허 자문 훈련용 가상 사안입니다. 지정된 관할과 시점 안에서 '
                '제공된 사실과 근거를 분석하고, 반대 논거·조건부 선택지·추가로 확인할 사실을 설명하세요. '
                '현재 법상태나 누락된 사실을 단정하지 마세요. JSON의 answer, citations(제공된 evidence_id 배열), '
                'limitations(제약사항 문자열 배열)로 답하세요.\n'+json.dumps({'jurisdiction':case['jurisdiction'],
                'temporal_scope':ep['temporal_scope'],'scenario_origin':ep['scenario_origin'],
                'question':ep['question'],'facts':ep['scenario_facts'],'evidence':context},ensure_ascii=False))
            policy[split].append({'episode_id':eid,'task_type':'patent_advisory','images':[],
                'messages':[{'role':'user','content':text}]})
            correct={'answer':ep['reference_answer'],'citations':ep['evidence_ids'],'limitations':[ep['temporal_scope']]}
            inferior={'answer':ep['inferior_answer'],'citations':ep['evidence_ids'],'limitations':[ep['temporal_scope']]}
            references[split].append({'episode_id':eid,'task_type':'patent_advisory','response':correct,
                'required_points':ep['required_points'],'forbidden_claims':ep['forbidden_claims'],
                'source_evidence':context,'review':proof})
            pairs[split].append({'pair_id':eid+'-P1','episode_id':eid,'category':'advisory_'+lane,
                'chosen':correct,'rejected':inferior,'preference_reason':ep['preference_reason'],
                'substantive_review':proof})
            registry.append({'episode_id':eid,'record_id':case['case_id'],'split':split,
                'task_type':'patent_advisory','jurisdiction':case['jurisdiction'],
                'source_record_sha256':canonical_hash(case),'lineage_key':info['lineage_key']})

    old_tasks={r['source_record_id']:r for split in ('train','validation')
               for r in load_rows(ROOT/f'data/case_rl/tasks.{split}.jsonl')}
    new_tasks={r['record_id']:r for r in load_rows(ROOT/'data/case_rl/new_claim_test/manifest.jsonl')}
    ext_path=ROOT/'rl_materials/curation/annotation_extensions.jsonl'
    extensions=load_rows(ext_path) if ext_path.is_file() else []
    by_drawing=defaultdict(list)
    for ext in extensions:by_drawing[ext['drawing_record_id']].append(ext)
    train_cards={r['card_id']:r for r in cards['train']}
    annotations_included=Counter()
    for lane,filename in [('drawings','claimsets.jsonl'),('new_test','claimsets_test.jsonl')]:
        for row in load_rows(ROOT/f'rl_materials/curation/{filename}'):
            eligible,proof=approved(row,lane)
            info=(lineage_new if lane=='new_test' else lineage_drawings).get(row['record_id'],{})
            if not info or (lane=='drawings' and info.get('record_sha256')!=canonical_hash(row)):
                eligible,proof=False,'drawing_lineage_missing_or_stale'
            if info.get('old_test_family_overlap') or info.get('validation_to_train_overlap') or info.get('protected_overlap') or info.get('case_overlap'):
                eligible,proof=False,'drawing_family_overlap'
            if not eligible:
                quarantined.append({'record_id':row['record_id'],'lane':lane,'reason':proof})
                continue
            task=(new_tasks if lane=='new_test' else old_tasks)[row['record_id']]
            split='test' if lane=='new_test' else ('calibration' if row['source_split']=='validation' else 'train')
            images=[]
            for p in row['images']:
                original=ROOT/p if lane=='new_test' else ROOT/'data/rl_source_v2'/p
                rel, provenance=asset(original)
                images.append(rel);image_sources[rel]=provenance
            eid='claimset-'+row['record_id']
            messages=[{'role':'user','content':CLAIM_INSTRUCTION}] if lane=='new_test' else task['prompt']
            language=task.get('output_language',task.get('language','ko'))
            jurisdiction=task.get('jurisdiction','unassigned')
            ext=None;ext_proof=None;provided=[]
            extension_rows=by_drawing.get(row['record_id'],[])
            if len(extension_rows)>1:
                raise ValueError('Duplicate drawing annotation extension:'+row['record_id'])
            if extension_rows:
                candidate=extension_rows[0]
                valid,receipt=approved(candidate,'annotation')
                errors=annotation_errors(candidate,row,train_cards,split,info['lineage_key'])
                if candidate.get('jurisdiction')!=jurisdiction:errors.append('annotation_task_jurisdiction')
                if valid and not errors:
                    ext,ext_proof=candidate,receipt
                    # Hide reviewer identity and gold-only metadata from policy context.
                    provided=[{k:v for k,v in train_cards[cid].items() if k not in {'review','review_status','lineage_key','case_patent_lookup_ids','usage'}} for cid in ext['card_ids']]
                    annotations_included[split]+=1
                else:
                    quarantined.append({'record_id':candidate['record_id'],'lane':'annotation','reason':errors or receipt})
            policy[split].append({'episode_id':eid,'task_type':'claim_set','images':images,
                'messages':messages+[{'role':'user','content':json.dumps({'output_language':language,
                    'jurisdiction':jurisdiction,'case_cards':provided},ensure_ascii=False)}]})
            correct=as_output(row['claims'],row['parents'],row['abstentions'])
            if ext:correct['annotations']=ext['annotations']
            references[split].append({'episode_id':eid,'task_type':'claim_set','response':correct,
                'required_points':row['reward_required_points']+(ext['required_points'] if ext else []),
                'forbidden_claims':row['reward_forbidden_claims']+(ext['forbidden_claims'] if ext else []),
                'observations':row['observations'],'claim_evidence':row['claim_evidence'],'review':proof,
                'annotation_review':ext_proof})
            for index, alt in enumerate(row['alternatives'],1):
                inferior=as_output(alt['claims'],alt['parents'],row['abstentions'])
                if ext:inferior['annotations']=ext['annotations']
                pairs[split].append({'pair_id':eid+f'-P{index}','episode_id':eid,'category':'drawing_structure',
                    'chosen':correct,'rejected':inferior,
                    'preference_reason':alt['preference_reason'],'defect':alt['defect'],'substantive_review':proof})
            if ext:
                inferior={**correct,'annotations':ext['inferior_annotations']}
                pairs[split].append({'pair_id':eid+'-ANNOTATION','episode_id':eid,'category':'case_annotation',
                    'chosen':correct,'rejected':inferior,'preference_reason':ext['preference_reason'],
                    'substantive_review':ext_proof})
            source_reviews.append(proof)
            registry.append({'episode_id':eid,'record_id':row['record_id'],'split':split,
                'source_split':row['source_split'],'task_type':'claim_set','jurisdiction':jurisdiction,
                'source_record_sha256':canonical_hash(row),'image_paths_in_source_order':row['images'],
                'lineage_key':info['lineage_key'],'annotation_record_id':ext['record_id'] if ext else None,
                'annotation_record_sha256':canonical_hash(ext) if ext else None})
    for split in ('train','calibration','test'):
        write_jsonl(OUT/f'policy/{split}.jsonl',policy[split])
        write_jsonl(OUT/f'reward/references.{split}.jsonl',references[split])
        write_jsonl(OUT/f'reward/preferences.{split}.jsonl',pairs[split])
        write_jsonl(OUT/f'evidence/cards.{split}.jsonl',cards[split])
    write_json(OUT/'audit/registry.json',registry)
    write_json(OUT/'audit/images.json',list(image_sources.values()))
    write_json(OUT/'audit/quarantined.json',quarantined)
    write_json(OUT/'audit/case_splits.json',all_case_splits)
    counts={s:{'policy_episodes':len(policy[s]),'preference_pairs':len(pairs[s]),
               'task_types':dict(Counter(r['task_type'] for r in policy[s]))} for s in ('train','calibration','test')}
    report={'status':'quarantined_candidate_not_a_training_release','materials_complete':False,
        'gpu_allowed':False,'model_quality':'unmeasured','counts':counts,'quarantined_records':len(quarantined),
        'case_split_conflicts':split_conflicts,'lineage_audit_passed':lineage.get('passed',False),
        'annotation_extensions_included':dict(annotations_included),'blind_calibration_passed':False,'processor_probe_passed':False,
        'note':'Acceptance is computed by the final release audit; assembly alone cannot authorize training.'}
    write_json(OUT/'ASSEMBLY_STATUS.json',report)
    protocol=OUT/'REWARD_PROTOCOL.md';shutil.copyfile(ROOT/'rl_materials/REWARD_PROTOCOL.md',protocol)
    members=['ASSEMBLY_STATUS.json','REWARD_PROTOCOL.md','audit/registry.json','audit/images.json',
             'audit/quarantined.json','audit/case_splits.json']
    for split in ('train','calibration','test'):
        members += [f'policy/{split}.jsonl',f'reward/references.{split}.jsonl',
                    f'reward/preferences.{split}.jsonl',f'evidence/cards.{split}.jsonl']
    files={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in members}
    # Assets are enumerated explicitly, so remnants of earlier candidate builds
    # do not silently become members of this candidate.
    files.update({r['path']:r['sha256'] for r in image_sources.values()})
    write_json(OUT/'manifest.json',{'files':files,'content_digest':canonical_hash(files)})
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    assemble()
