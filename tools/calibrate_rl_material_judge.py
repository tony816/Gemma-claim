"""Prepare blind human/agent judge packets and score submitted decisions.

This script invokes no model. The judge is a separate, explicitly instructed
process. Labels and author preference reasons are never in its blind packet.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.validate_rl_curation import canonical_hash, load_rows
from tools.assemble_rl_release import OUT, write_json, write_jsonl, read_json

DIMENSIONS={
    'claim_set':{'image_support':4,'dependency_scope':3,'source_annotation':2,'uncertainty':1},
    'patent_advisory':{'fact_application':4,'evidence_fidelity':3,'jurisdiction_era':2,'decision_usefulness':1},
}


def scored_candidate(value,task_type):
    maxima=DIMENSIONS[task_type]
    if not isinstance(value,dict) or type(value.get('severe_error')) is not bool:
        return None
    dims=value.get('dimensions',{})
    if set(dims)!=set(maxima):return None
    if any(type(dims[k]) not in (int,float) or not 0<=dims[k]<=maximum for k,maximum in maxima.items()):
        return None
    score=sum(dims.values())/10
    return min(score,.25) if value['severe_error'] else score


def prepare(round_name, seed):
    directory=ROOT/'.superloopy/evidence/judge_calibration'/round_name
    if directory.exists():
        raise ValueError('Calibration round already exists; preserve it and use a new round name.')
    pairs=load_rows(OUT/'reward/preferences.calibration.jsonl')
    if len(pairs)<20:
        raise ValueError(f'Need at least20 reviewed held-out comparisons; have{len(pairs)}.')
    prompts={r['episode_id']:r for r in load_rows(OUT/'policy/calibration.jsonl')}
    refs={r['episode_id']:r for r in load_rows(OUT/'reward/references.calibration.jsonl')}
    randomizer=random.Random(seed)
    order=list(range(len(pairs)));randomizer.shuffle(order)
    blinded,labels=[],[]
    for count,index in enumerate(order,1):
        pair=pairs[index];eid=pair['episode_id'];prompt=prompts[eid];ref=refs[eid]
        side=randomizer.choice(['A','B'])
        candidate_a=pair['chosen'] if side=='A' else pair['rejected']
        candidate_b=pair['rejected'] if side=='A' else pair['chosen']
        blind={'comparison_id':f'C{count:03d}', 'task_type':prompt['task_type'],
            'messages':prompt['messages'],
            'images':[str((OUT/p).resolve()) for p in prompt['images']],
            'rubric':{'required_points':ref['required_points'],'forbidden_claims':ref['forbidden_claims']},
            'candidate_A':candidate_a,'candidate_B':candidate_b}
        blind['input_sha256']=canonical_hash(blind)
        blinded.append(blind)
        labels.append({'comparison_id':blind['comparison_id'],'expected':side,
            'source_pair_id':pair['pair_id'],'source_episode_id':eid,'category':pair['category'],
            'pair_sha256':canonical_hash(pair),'input_sha256':blind['input_sha256']})
    directory.mkdir(parents=True)
    write_jsonl(directory/'blind.jsonl',blinded)
    write_json(directory/'labels.PRIVATE.json',labels)
    protocol=ROOT/'rl_materials/REWARD_PROTOCOL.md'
    write_json(directory/'preparation.json',{'round':round_name,'seed':seed,'comparisons':len(blinded),
        'protocol_path':str(protocol),'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'blind_sha256':hashlib.sha256((directory/'blind.jsonl').read_bytes()).hexdigest(),
        'labels_sha256':hashlib.sha256((directory/'labels.PRIVATE.json').read_bytes()).hexdigest(),
        'gold_source':'exact-hash independent semantic acceptance',
        'judge_may_read':['blind.jsonl',str(protocol),'only image paths named in blind.jsonl'],
        'judge_must_not_read':['labels.PRIVATE.json','curation records','reward references/preferences','review findings'],
        'scope':'Native independent judge protocol on these authored comparisons; no future API/GPU judge backend certified.'})
    print(json.dumps({'directory':str(directory),'comparisons':len(blinded),'judged':False}))


def score(round_name):
    directory=ROOT/'.superloopy/evidence/judge_calibration'/round_name
    prep=read_json(directory/'preparation.json')
    labels=read_json(directory/'labels.PRIVATE.json')
    blind=load_rows(directory/'blind.jsonl')
    submission=read_json(directory/'decisions.json')
    errors=[]
    for filename,key in [('blind.jsonl','blind_sha256'),('labels.PRIVATE.json','labels_sha256')]:
        if hashlib.sha256((directory/filename).read_bytes()).hexdigest()!=prep[key]:
            errors.append('calibration_source_changed:'+filename)
    if hashlib.sha256(Path(prep['protocol_path']).read_bytes()).hexdigest()!=prep['protocol_sha256']:
        errors.append('reward_protocol_changed_after_blinding')
    if not submission.get('judge') or submission.get('gold_labels_read') is not False:
        errors.append('judge_identity_or_blinding_attestation_missing')
    decisions=submission.get('decisions',[])
    if len(decisions)!=len(labels) or len({r.get('comparison_id') for r in decisions})!=len(decisions):
        errors.append('missing_or_duplicate_decisions')
    lookup={r.get('comparison_id'):r for r in decisions}
    blind_lookup={r['comparison_id']:r for r in blind}
    category=defaultdict(lambda:{'total':0,'correct':0,'errors':[]})
    pair_results=[]
    for label in labels:
        decision=lookup.get(label['comparison_id'],{})
        task_type=blind_lookup[label['comparison_id']]['task_type']
        score_a=scored_candidate(decision.get('score_A'),task_type)
        score_b=scored_candidate(decision.get('score_B'),task_type)
        valid=(decision.get('input_sha256')==label['input_sha256'] and decision.get('choice') in {'A','B','tie','uncertain'}
               and bool(decision.get('reason')) and bool(decision.get('evidence_checked'))
               and score_a is not None and score_b is not None)
        if valid and decision['choice'] in {'A','B'}:
            valid=(score_a>score_b) if decision['choice']=='A' else (score_b>score_a)
        if not valid:
            errors.append('invalid_or_unbound_judge_decision:'+label['comparison_id'])
        correct=valid and decision.get('choice')==label['expected']
        group=category[label['category']];group['total']+=1;group['correct']+=int(correct)
        if not correct:group['errors'].append(label['comparison_id'])
        pair_results.append({'comparison_id':label['comparison_id'],'category':label['category'],
            'correct':correct,'expected':label['expected'],'observed':decision.get('choice'),
            'score_A':score_a,'score_B':score_b,
            'input_sha256':label['input_sha256']})
    total=len(labels);correct=sum(r['correct'] for r in pair_results)
    report={'passed':not errors and total>=20 and correct/total>=.85,
        'comparisons':total,'correct':correct,'agreement':correct/total if total else 0,
        'threshold':.85,'judge':submission.get('judge'),'judge_configuration':submission.get('configuration'),
        'protocol_sha256':prep['protocol_sha256'],'round':round_name,'errors':errors,
        'categories':dict(category),'pair_results':pair_results,
        'decisions_sha256':hashlib.sha256((directory/'decisions.json').read_bytes()).hexdigest(),
        'blind_sha256':prep['blind_sha256'],'labels_sha256':prep['labels_sha256'],
        'scope':prep['scope'],'model_quality_measured':False,'gpu_allowed':False}
    write_json(directory/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='pair_results'},ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','score'])
    parser.add_argument('--round',default='round1')
    parser.add_argument('--seed',type=int,default=20260908)
    args=parser.parse_args()
    if not args.round.replace('_','').isalnum():raise ValueError('Use a simple round name')
    if args.action=='prepare':prepare(args.round,args.seed)
    else:raise SystemExit(0 if score(args.round)['passed'] else 1)
