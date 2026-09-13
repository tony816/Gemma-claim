"""Mechanical source bindings for annotation comparisons, not semantic approval."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.validate_rl_curation import canonical_hash,load_rows
from tools.assemble_rl_release import annotation_errors,approved,case_partition,read_json,write_json


def audit():
    lineage=read_json(ROOT/'data/case_rl/lineage_audit.json',{})
    case_info={r['case_id']:r for r in lineage.get('case_lineages',[])}
    drawing_info={r['record_id']:r for key in ('selected_drawings','new_test') for r in lineage.get(key,[])}
    cards={}
    for lane in ('kr','us'):
        for i,case in enumerate(load_rows(ROOT/f'rl_materials/curation/{lane}_cases.jsonl')):
            valid,proof=approved(case,lane)
            info=case_info.get(case['case_id'],{})
            if valid and case_partition(case,i)=='train' and info.get('record_sha256')==canonical_hash(case):
                for p in case['principles']:
                    cards[p['id']]={'case_id':case['case_id'],'jurisdiction':case['jurisdiction'],
                        'review':proof,'review_status':'approved','lineage_key':info['lineage_key']}
    drawings={r['record_id']:r for name in ('claimsets.jsonl','claimsets_test.jsonl')
              for r in load_rows(ROOT/f'rl_materials/curation/{name}')}
    tasks={r['source_record_id']:r for split in ('train','validation')
           for r in load_rows(ROOT/f'data/case_rl/tasks.{split}.jsonl')}
    tasks.update({r['record_id']:r for r in load_rows(ROOT/'data/case_rl/new_claim_test/manifest.jsonl')})
    rows=load_rows(ROOT/'rl_materials/curation/annotation_extensions.jsonl')
    results=[];errors=[]
    for r in rows:
        rid=r['record_id'];did=r['drawing_record_id'];drawing=drawings.get(did)
        if drawing is None:
            issues=['drawing_missing']
        else:
            issues=annotation_errors(r,drawing,cards,r['source_split'],drawing_info.get(did,{}).get('lineage_key'))
            valid,_=approved(drawing,'new_test' if r['source_split']=='new_test' else 'drawings')
            if not valid:issues.append('drawing_not_accepted')
            if r.get('jurisdiction')!=tasks[did].get('jurisdiction'):issues.append('task_jurisdiction')
        errors += [rid+':'+x for x in issues]
        results.append({'record_id':rid,'sha256':canonical_hash(r),'errors':issues})
    if len({r['record_id'] for r in rows})!=len(rows) or len({r['drawing_record_id'] for r in rows})!=len(rows):
        errors.append('duplicate_annotation_or_drawing_id')
    report={'passed':not errors,'records':len(rows),'source_splits':dict(Counter(r['source_split'] for r in rows)),
        'errors':errors,'record_hashes':results,'semantic_review_passed':False}
    write_json(ROOT/'.superloopy/evidence/annotation-source-schema-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='record_hashes'},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(0 if audit()['passed'] else 1)
