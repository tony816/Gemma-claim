"""Gated replacement of the existing endpoint and independent bounded rollback."""
import json
import os
import re
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi

from tools.rl_cloud import runpod, graphql, credentials
from .core import ROOT, OUT, CONFIG, write, append

ENDPOINT = 'fdiltabt78bogm'
ROLLBACK = ROOT/'run_artifacts/rl_v3_20260908/endpoint_v2_rollback.json'


def redact(value):
    text = json.dumps(value, ensure_ascii=False)
    for key in ('HF_TOKEN', 'RUNPOD_API_KEY'):
        secret = os.environ.get(key)
        if secret:
            text = text.replace(secret, '[REDACTED]')
    text=re.sub(r'\bhf_[A-Za-z0-9]{12,}\b','[REDACTED_HF_TOKEN]',text)
    text=re.sub(r'\brpa_[A-Za-z0-9_-]{12,}\b','[REDACTED_RUNPOD_TOKEN]',text)
    return json.loads(text)


def public_config(config):
    result = {k:v for k,v in config.items() if k != 'env'}
    result['env'] = {k:v for k,v in config.get('env', {}).items()
                     if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RUNPOD_API_KEY'}}
    result['secret_env_keys'] = [k for k in config.get('env', {})
                               if k in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RUNPOD_API_KEY'}]
    return redact(result)


def data_api(route, payload=None):
    credentials()
    response = requests.request('GET' if payload is None else 'POST',
        f'https://api.runpod.ai/v2/{ENDPOINT}/{route}', json=payload,
        headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}, timeout=30)
    assert response.ok, f'data_api_http_{response.status_code}'
    return response.json()


def private_template(template_id):
    response = graphql('query { myself { podTemplates { id isPublic } } }')
    assert not response.get('errors')
    found = [t for t in response['data']['myself']['podTemplates'] if t['id'] == template_id]
    assert len(found) == 1 and found[0]['isPublic'] is False
    return True


def snapshot():
    account = graphql('query { myself { isAutoPayEnabled clientBalance currentSpendPerHr '
        'endpoint(id:"'+ENDPOINT+'") { id pods { id desiredStatus runtime { uptimeInSeconds } } } } }')
    assert not account.get('errors'), 'graphql_state_unavailable'
    return {'time':time.time(), 'config':public_config(runpod('/serverless/'+ENDPOINT)),
        'workers':runpod('/serverless/'+ENDPOINT+'/workers'),
        'active_pods':[p for p in account['data']['myself']['endpoint']['pods']
                       if p['desiredStatus'] not in {'EXITED','TERMINATED'}],
        'health_not_polled':'RunPod documents that health reads trigger worker Sync; use control-plane desiredStatus for shutdown',
        'account':account['data']['myself']}


def logs():
    credentials()
    workers = runpod('/serverless/'+ENDPOINT+'/workers')
    collected = []
    for worker in workers.get('workers',[]):
        worker_id = worker['id']
        started = time.monotonic()
        lines = []
        try:
            with requests.get(f'https://api.runpod.io/v2/serverless/{ENDPOINT}/workers/{worker_id}/logs?tail=300',
                headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],
                         'User-Agent':'Mozilla/5.0'},stream=True,timeout=(10,5)) as response:
                assert response.ok, f'worker_log_http_{response.status_code}'
                for line in response.iter_lines():
                    if line: lines.append(line.decode('utf-8','replace'))
                    if len(lines)>=600 or time.monotonic()-started > 6: break
        except requests.RequestException:
            pass  # A quiet SSE stream ends via read timeout; keep received lines.
        value = redact({'time':time.time(),'worker_id':worker_id,'worker':worker,'lines':lines})
        append('serving/worker_logs.jsonl',value)
        collected.append(value)
    print(json.dumps(collected,ensure_ascii=False)[-18000:])


def rollback(reason):
    result=runpod('/endpoints/'+ENDPOINT,'PATCH',{
        'templateId':'99icte8riw','workersMin':0,'workersMax':0,'idleTimeout':5,
        'flashboot':False,'gpuCount':1,'gpuTypeIds':['NVIDIA A40','NVIDIA RTX A6000'],
        'networkVolumeIds':[]},v1=True)
    current=runpod('/serverless/'+ENDPOINT)
    assert current['env']['MODEL_NAME']=='cyankiwi/gemma-4-31B-it-qat-AWQ-INT4'
    if current['env'].get('HF_TOKEN'):
        env=dict(current['env']);env['HUGGING_FACE_HUB_TOKEN']=env.pop('HF_TOKEN')
        current=runpod('/serverless/'+ENDPOINT,'PATCH',{'env':env,'workers':{'min':0,'max':0,'idleTimeout':5}})
    write('serving/rollback.json',{'time':time.time(),'reason':reason,
        'config':public_config(current),'v2_revision':CONFIG['v2_revision']})


def deploy():
    credentials()
    proof=json.loads((OUT/'pod/fresh_reload_verified.json').read_text())
    ended=json.loads((OUT/'termination.json').read_text())
    hashes=json.loads((OUT/'pod/model_hashes.json').read_text())
    assert proof['eligible_to_deploy'] and proof['local_files_only_adapter']
    assert json.loads((OUT/'comparison.json').read_text())['corrected_calibration_gate']['passed']
    assert proof['actual_adapter_tensor_match'] and proof['hashes_verified']==len(hashes)
    assert ended['pod_absent'] and not runpod('/pods',v1=True)
    info=HfApi().model_info(proof['repo'],revision=proof['commit'])
    assert info.private and info.sha==proof['commit'] and proof['repo']==CONFIG['output_repo']
    assert json.loads((OUT/'serving/cpu_bootstrap.json').read_text())['passed']
    access=json.loads((OUT/'serving/hub_auth_access.json').read_text())
    assert access['passed']
    heartbeat=json.loads((OUT/'serving/guard_armed.json').read_text())
    assert time.time()-heartbeat['time']<60,'serving_guard_not_live'
    assert not (OUT/'serving/started.json').exists(),'refuse_duplicate_deployment'
    current=runpod('/serverless/'+ENDPOINT)
    assert current['workers']['min']==0 and current['workers']['max']==0
    before=snapshot()
    budget=json.loads((OUT/'additional_budget.json').read_text())
    spent=budget['observed_balance_after_topup']-before['account']['clientBalance']
    assert not before['account']['isAutoPayEnabled']
    assert before['account']['currentSpendPerHr']==0
    assert spent+1.0<=budget['additional_execution_cap_usd'],'remaining_authorized_budget_below_serverless_reserve'
    write('serving/budget_admission.json',{'observed_additional_balance_debit':spent,
        'serverless_reserve_usd':1.,'additional_cap_usd':budget['additional_execution_cap_usd']})
    write('serving/before.json',before)
    # worker-vllm maps HF_TOKEN to a CLI flag and logs that command. The Hub's
    # supported legacy env alias authenticates without entering the CLI builder.
    hub_token=current['env'].get('HF_TOKEN') or current['env']['HUGGING_FACE_HUB_TOKEN']
    if access['selected_token_source']=='project_environment':hub_token=os.environ['HF_TOKEN']
    assert HfApi(token=hub_token).model_info(proof['repo'],revision=proof['commit']).private
    env={'HUGGING_FACE_HUB_TOKEN':hub_token,'HF_HOME':'/tmp/hf',
        'HUGGINGFACE_HUB_CACHE':'/tmp/hf/hub','HF_HUB_CACHE':'/tmp/hf/hub',
        'HF_DATASETS_CACHE':'/tmp/hf/datasets','BASE_PATH':'/tmp',
        'MODEL_NAME':CONFIG['base_model'],'MODEL_REVISION':CONFIG['base_revision'],
        'TOKENIZER_REVISION':CONFIG['base_revision'],'OPENAI_SERVED_MODEL_NAME_OVERRIDE':'gemma4-31b',
        'CLAIM_ADAPTER_REPO':proof['repo'],'CLAIM_ADAPTER_REVISION':proof['commit'],
        'CLAIM_ADAPTER_HASHES':json.dumps(hashes),'ENABLE_LORA':'true','MAX_LORAS':'1',
        'MAX_LORA_RANK':'16','MAX_MODEL_LEN':'8192','MAX_NUM_SEQS':'1','MAX_CONCURRENCY':'1','DTYPE':'bfloat16',
        'GPU_MEMORY_UTILIZATION':'0.95','ENABLE_PREFIX_CACHING':'true','TRUST_REMOTE_CODE':'false',
        'RAW_OPENAI_OUTPUT':'true','VLLM_STARTUP_TIMEOUT':'650','TOKENIZERS_PARALLELISM':'false'}
    started_at=time.time()
    env.update(RUNPOD_API_KEY=os.environ['RUNPOD_API_KEY'],CLAIM_DEPLOYMENT_DEADLINE=str(started_at+1200))
    code=(ROOT/'serving/pinned_lora_bootstrap.py').read_text(encoding='utf-8')
    template_payload={
        'name':'gemma-claim-v3-pinned-'+proof['commit'][:8],'imageName':current['image'],
        'isPublic':False,'isServerless':True,'category':'NVIDIA','containerDiskInGb':100,
        'volumeInGb':0,'ports':[],'dockerEntrypoint':['python3','-c',code],
        'dockerStartCmd':[],'env':env}
    existing=[t for t in runpod('/templates',v1=True) if t.get('name')==template_payload['name']]
    assert len(existing)<=1,'ambiguous_existing_template'
    if existing:
        private_template(existing[0]['id'])
        patch_payload={k:v for k,v in template_payload.items() if k not in {'category','isServerless'}}
        template=runpod('/templates/'+existing[0]['id'],'PATCH',patch_payload,v1=True)
    else:
        template=runpod('/templates','POST',template_payload,v1=True)
    private_template(template['id'])
    write('serving/template_created.json',{'time':time.time(),'id':template['id'],
        'name':template['name'],'image':template['imageName'],'private':True,
        'model_revision':proof['commit'],'bootstrap_sha256':__import__('hashlib').sha256(code.encode()).hexdigest()})
    # Written before paid mutation so an uncertain response cannot evade rollback.
    write('serving/started.json',{'time':started_at,'max_seconds':1200,
        'estimated_max_compute_usd':1200/3600*2.72,'model':proof['repo'],'revision':proof['commit']})
    runpod('/endpoints/'+ENDPOINT,'PATCH',{'templateId':template['id'],
        'workersMin':0,'workersMax':1,'idleTimeout':5,'flashboot':False,'gpuCount':1,
        'gpuTypeIds':['NVIDIA A100-SXM4-80GB','NVIDIA A100 80GB PCIe'],
        'networkVolumeIds':[],'executionTimeoutMs':600000},v1=True)
    result=runpod('/serverless/'+ENDPOINT)
    assert result['env']['CLAIM_ADAPTER_REVISION']==proof['commit']
    assert result['env']['MODEL_NAME']==CONFIG['base_model']
    write('serving/deployed.json',{'time':time.time(),'config':public_config(result),'template_id':template['id']})


def guard():
    while True:
        try:
            write('serving/guard_armed.json', {'time':time.time(),'pid':os.getpid()})
            if (OUT/'serving/verified_complete.json').exists():
                return
            started = OUT/'serving/started.json'
            if started.exists():
                record = json.loads(started.read_text())
                if time.time() >= record['time']+record['max_seconds']:
                    rollback('independent_serverless_deadline')
                    final = snapshot()
                    write('serving/guard_finished.json', final)
                    if final['config']['workers']['max'] == 0:
                        return
        except Exception as exc:
            append('serving/guard_errors.jsonl', {'time':time.time(),'type':type(exc).__name__})
        time.sleep(10)


def disarm_verified():
    proof=json.loads((OUT/'serving/ui_smoke.json').read_text())
    assert proof['passed']
    state=snapshot()
    assert not state['active_pods'],'endpoint_gpu_pods_still_present'
    assert state['account']['currentSpendPerHr']==0
    template=json.loads((OUT/'serving/template_created.json').read_text())
    private=runpod('/templates/'+template['id'],v1=True)
    private_template(template['id'])
    env=dict(private['env']);env.pop('RUNPOD_API_KEY',None);env.pop('CLAIM_DEPLOYMENT_DEADLINE',None)
    runpod('/templates/'+template['id'],'PATCH',{'env':env},v1=True)
    current=runpod('/serverless/'+ENDPOINT)
    assert 'RUNPOD_API_KEY' not in current['env'] and 'CLAIM_DEPLOYMENT_DEADLINE' not in current['env']
    write('serving/guard_disarmed.json',{'time':time.time(),'reason':'UI tests passed and actual GPU Pods zero',
        'control_key_removed_from_template':True,'template_id':template['id']})


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'guard': guard()
    elif sys.argv[1] == 'deploy': deploy()
    elif sys.argv[1] == 'snapshot':
        value = snapshot(); append('serving/observations.jsonl',value); print(json.dumps(value))
    elif sys.argv[1] == 'logs': logs()
    elif sys.argv[1] == 'rollback': rollback('operator_deployment_failure')
    elif sys.argv[1] == 'disarm': disarm_verified()
