"""CPU-only encoding audit of the actual assembled policy/reward material.

Loads only the cached processor/tokenizer. It verifies generation input encoding
and separately encoded reference continuations. This is not a model evaluation,
an SFT run, or validation of an online RL trainer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'pipeline'))
from tools.validate_rl_curation import canonical_hash,load_rows
from tools.assemble_rl_release import OUT,write_json


def main(sample=False):
    import torch
    from transformers import AutoProcessor
    from common import normalise_record,build_chat_messages
    from dataset import encode_record
    torch.set_num_threads(2)
    revision='842da3794eaa0b77d5f08bae87a17459d91ff475'
    processor=AutoProcessor.from_pretrained('google/gemma-4-31B-it',revision=revision,local_files_only=True)
    rows=[]
    for split in ('train','calibration','test'):
        refs={r['episode_id']:r for r in load_rows(OUT/f'reward/references.{split}.jsonl')}
        rows += [(split,p,refs[p['episode_id']]) for p in load_rows(OUT/f'policy/{split}.jsonl')]
    total=len(rows)
    if not rows:raise ValueError('No accepted policy episodes to encode')
    if sample:
        rows=list({r[1]['episode_id']:r for r in [max(rows,key=lambda r:len(r[1]['images'])),
             max(rows,key=lambda r:len(json.dumps(r[1],ensure_ascii=False))) ]}.values())
    results=[]
    for index,(split,prompt,ref) in enumerate(rows):
        response=json.dumps(ref['response'],ensure_ascii=False,separators=(',',':'))
        raw={'id':prompt['episode_id'],'images':prompt['images'],
             'messages':prompt['messages']+[{'role':'assistant','content':response}]}
        record=normalise_record(raw,index,split,OUT)
        prompt_messages=build_chat_messages(record,for_prompt=True)
        if not all(m['role']!='assistant' for m in prompt_messages):
            raise ValueError('Assistant reference leaked into generation input: '+prompt['episode_id'])
        placeholders=sum(p.get('type')=='image' for m in prompt_messages for p in m['content'])
        if placeholders!=len(prompt['images']):
            raise ValueError('Image placeholder count mismatch: '+prompt['episode_id'])
        encoded=encode_record(processor,record)
        if not bool((encoded['labels'][0,:encoded['_prompt_len']]==-100).all()):
            raise ValueError('Prompt labels are unmasked: '+prompt['episode_id'])
        if not bool((encoded['labels'][0,encoded['_prompt_len']:]!=-100).any()):
            raise ValueError('Reference continuation is fully masked: '+prompt['episode_id'])
        results.append({'episode_id':prompt['episode_id'],'task_type':prompt['task_type'],
            'policy_sha256':canonical_hash(prompt),'reward_reference_sha256':canonical_hash(ref),
            'image_count':len(prompt['images']),'prompt_tokens':encoded['_prompt_len'],
            'reference_tokens':encoded['_target_len'],'total_tokens':encoded['_total_len'],
            'policy_contains_no_assistant_reference':True,'image_placeholders_match':True,
            'assistant_reference_mask_verified':True,'truncation':False})
        if (index+1)%20==0:print(f'CPU material encoding {index+1}/{len(rows)}',flush=True)
    report={'passed':not sample and len(results)==total,'sample_only':sample,'records':results,
        'total_candidate_episodes':total,'encoded_episodes':len(results),'processor_revision':revision,
        'max_prompt_tokens':max(r['prompt_tokens'] for r in results),
        'max_total_tokens':max(r['total_tokens'] for r in results),
        'model_weights_loaded':False,'device':'cpu','gpu_allowed':False,
        'online_rl_runtime_verified':False,'model_quality_measured':False}
    target=ROOT/'.superloopy/evidence'/('release-processor-sample.json' if sample else 'release-processor-probe.json')
    write_json(target,report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample',action='store_true')
    main(parser.parse_args().sample)
