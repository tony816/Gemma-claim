"""RunPod template entrypoint, embedded verbatim; credentials remain in env."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import time
from huggingface_hub import snapshot_download


def arm_deployment_guard():
    deadline=os.environ.get('CLAIM_DEPLOYMENT_DEADLINE')
    if not deadline:return
    # Initial rollout only. A verified idle deployment removes these env keys.
    # The process remains inside this worker when main.py replaces our process.
    import subprocess
    code='''import json,os,time,urllib.request
deadline=os.environ['CLAIM_DEPLOYMENT_DEADLINE']
while time.time()<float(deadline):time.sleep(max(0,min(5,float(deadline)-time.time())))
url='https://api.runpod.io/v2/serverless/fdiltabt78bogm'
headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'Content-Type':'application/json'}
while True:
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=20) as r:state=json.load(r)
  if state.get('env',{}).get('CLAIM_DEPLOYMENT_DEADLINE')!=deadline:break
  payload=json.dumps({'workers':{'min':0,'max':0,'idleTimeout':5}}).encode()
  with urllib.request.urlopen(urllib.request.Request(url,data=payload,headers=headers,method='PATCH'),timeout=20) as r:assert r.status==200
  print('CLAIM_DEPLOYMENT_DEADLINE_PAUSED_ENDPOINT',flush=True);break
 except Exception:time.sleep(5)
'''
    subprocess.Popen([sys.executable,'-c',code],stdin=subprocess.DEVNULL)
    print('CLAIM_DEPLOYMENT_GUARD_ARMED '+deadline,flush=True)
    assert time.time()<float(deadline),'deployment_deadline_expired'


def prepare(target=None):
    # Never allow the worker's generic CLI builder to emit --hf-token in logs.
    if os.environ.get('HF_TOKEN'):
        os.environ['HUGGING_FACE_HUB_TOKEN']=os.environ.pop('HF_TOKEN')
    repo=os.environ['CLAIM_ADAPTER_REPO'];revision=os.environ['CLAIM_ADAPTER_REVISION']
    expected=json.loads(os.environ['CLAIM_ADAPTER_HASHES'])
    assert len(revision)==40 and 'adapter_model.safetensors' in expected
    target=Path('/tmp/claim-v3') if target is None else Path(target)
    snapshot_download(repo,revision=revision,allow_patterns=list(expected),local_dir=target)
    for name,value in expected.items():
        path=(target/name).resolve();assert path.is_relative_to(target.resolve())
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==value
    config=json.loads((target/'adapter_config.json').read_text())
    assert config['r']==16 and config['base_model_name_or_path']==os.environ['MODEL_NAME']
    assert config['revision']==os.environ['MODEL_REVISION']
    assert all(x.startswith('model.language_model.layers.') for x in config['target_modules'])
    assert importlib.metadata.version('vllm')=='0.28.0'
    module={'name':'claim-v3','path':str(target),'base_model_name':'gemma4-31b'}
    args=['--revision',os.environ['MODEL_REVISION'],'--tokenizer-revision',os.environ['MODEL_REVISION'],
          '--lora-modules',json.dumps(module),'--enforce-eager','--limit-mm-per-prompt',
          json.dumps({'image':8,'video':0,'audio':0})]
    os.environ['VLLM_EXTRA_ARGS']=shlex.join(args)
    print('CLAIM_PINNED_ADAPTER_VERIFIED '+json.dumps({'repo':repo,'revision':revision,
        'adapter_sha256':expected['adapter_model.safetensors'],'base_revision':config['revision'],
        'rank':config['r'],'targets':len(config['target_modules']),'vllm':'0.28.0'}),flush=True)
    return args


if __name__=='__main__':
    arm_deployment_guard()
    prepare()
    os.execv(sys.executable,[sys.executable,'/src/main.py'])
