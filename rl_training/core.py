import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'rl_training/config.json').read_text())
RELEASE = ROOT/CONFIG['release']
OUT = Path(os.environ.get('RL_OUT', str(ROOT/CONFIG['artifact_directory'])))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def append(name, value):
    path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+'\n')


def rows(relative):
    return [json.loads(s) for s in (RELEASE/relative).read_text(encoding='utf-8').splitlines() if s.strip()]


def verify_release():
    assert sha(RELEASE/'manifest.json') == CONFIG['manifest_sha256']
    manifest = json.loads((RELEASE/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (RELEASE/name).resolve()
        assert path.is_relative_to(RELEASE.resolve()) and sha(path) == expected, name
    return {'files': len(manifest['files']), 'sha256': CONFIG['manifest_sha256']}


def parse_json(text):
    text = text.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
    return json.loads(fenced.group(1) if fenced else text)


def policy_inputs(processor, policy):
    from PIL import Image
    assert set(policy) == {'episode_id','task_type','messages','images'}
    messages = []
    for msg in policy['messages']:
        assert msg['role'] in {'user','system'} and isinstance(msg['content'], str)
        messages.append({'role':msg['role'], 'content':[{'type':'text','text':msg['content']}]})
    images = []
    for name in policy['images']:
        path=(RELEASE/name).resolve()
        assert path.is_relative_to(RELEASE.resolve())
        with Image.open(path) as image: images.append(image.convert('RGB').copy())
    if images:
        user=next(m for m in messages if m['role']=='user')
        user['content']=[{'type':'image'} for _ in images]+user['content']
    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs={'text':text,'return_tensors':'pt','padding':False}
    if images:kwargs['images']=images
    encoded=processor(**kwargs)
    assert encoded['input_ids'].shape[-1]+CONFIG['max_new_tokens'] <= 8192
    return encoded


def advantages(rewards):
    import torch
    values=torch.tensor(rewards,dtype=torch.float32)
    return (values-values.mean())/(values.std(unbiased=False)+1e-6)


def completion_inputs(prompt, sequence):
    import torch
    length=sequence.shape[-1]
    plen=prompt['input_ids'].shape[-1]
    assert length>plen and torch.equal(sequence[:,:plen],prompt['input_ids'])
    result=dict(prompt)
    result['input_ids']=sequence
    for name in ('attention_mask','mm_token_type_ids','token_type_ids'):
        if name in result:
            fill=1 if name=='attention_mask' else 0
            result[name]=torch.cat([result[name],torch.full((1,length-plen),fill,
                dtype=result[name].dtype,device=result[name].device)],dim=-1)
    return result


def rollout_loss(logits, target_tokens, advantage):
    import torch.nn.functional as F
    # The first retained position is the last prompt token, predicting the first
    # generated token. Prompt/image tokens never become training targets.
    assert logits.shape[:2] == (target_tokens.shape[0],target_tokens.shape[1]+1)
    log_probs=-F.cross_entropy(logits[:,:-1,:].float().reshape(-1,logits.shape[-1]),
                              target_tokens.reshape(-1),reduction='none')
    return -float(advantage)*log_probs.mean()


def shuffled_train():
    result=rows('policy/train.jsonl'); assert len(result)==88
    random.Random(CONFIG['seed']).shuffle(result)
    # First paid smoke includes an image, without changing the split.
    first=next(i for i,p in enumerate(result) if p['task_type']=='claim_set')
    result[0],result[first]=result[first],result[0]
    return result
