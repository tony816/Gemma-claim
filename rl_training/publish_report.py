"""Upload allowlisted, secret-scanned final evidence; never alter model weights."""
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from tools.rl_cloud import credentials
from .core import ROOT, OUT, CONFIG, write


def run():
    credentials()
    audit=json.loads((OUT/'closing_audit.json').read_text(encoding='utf-8'))
    assert not audit['pods'] and not audit['network_volumes']
    assert not audit['serverless']['active_pods']
    assert audit['serverless']['account']['isAutoPayEnabled'] is False
    assert audit['serverless']['account']['currentSpendPerHr']==0
    proof=json.loads((OUT/'pod/fresh_reload_verified.json').read_text(encoding='utf-8'))
    files=[ROOT/'RL_RESULTS.md',ROOT/'HANDOFF.md']
    for name in ('comparison.json','manual_test_review.json','manual_train10_review.json',
                 'closing_audit.json','termination.json','additional_budget.json',
                 'validated_model_tag.json','recovered_judge_response.json',
                 'evaluation_resolver_correction.json'):
        files.append(OUT/name)
    for name in ('fresh_reload_verified.json','hub_upload.json','model_hashes.json',
                 'recovery_scope.json','runtime.json','selected_loaded.json','v2_loaded.json',
                 'reference_unchanged.json','release_verified.json','selection.json'):
        files.append(OUT/'pod'/name)
    files.extend(sorted((OUT/'pod/evaluation').glob('*.jsonl')))
    for name in ('ui_smoke.json','manual_ui_review.json','ui_visual_check.json',
                 'cpu_tests_after_v3.json','worker_logs.jsonl','observations.jsonl',
                 'deployed.json','template_created.json','guard_disarmed.json',
                 'verified_complete.json','rollback.json','failed_deployment.json'):
        path=OUT/'serving'/name
        if path.exists():files.append(path)
    for name in ('template_removed.json','cleanup_verified.json','paused_job_status.json',
                 'paused_ui_visual_check.json','paused_ui_live_api_check.json'):
        path=OUT/'serving'/name
        if path.exists():files.append(path)
    files.extend(OUT/name for name in ('final_result.json','sleep_prevention_released.json'))
    files.extend(sorted((ROOT/'rl_training').glob('*.py')))
    files.extend(ROOT/path for path in ('serving/pinned_lora_bootstrap.py','serving/claim_client.py',
        'serving/rl_client.py','serving/launch_test.py','space/test_ui.py','space/rl_test_panel.py',
        'space/claim_client.py','serving/deployment_status.json','start-test.cmd','tests/test_serving_v2.py','tests/test_rl_execution_gates.py'))
    operations=[];manifest={}
    for path in files:
        blob=path.read_bytes()
        for key in ('HF_TOKEN','RUNPOD_API_KEY'):
            secret=os.environ.get(key,'')
            assert not secret or secret.encode() not in blob,f'secret_in_{path.name}'
        remote='operator-report-20260909/'+str(path.relative_to(ROOT)).replace('\\','/')
        operations.append(CommitOperationAdd(path_in_repo=remote,path_or_fileobj=blob))
        manifest[remote]=hashlib.sha256(blob).hexdigest()
    report=(ROOT/'RL_RESULTS.md').read_text(encoding='utf-8')
    card='---\nbase_model: google/gemma-4-31B-it\nlibrary_name: peft\ntags:\n- reinforcement-learning\n- lora\n- gemma4\n---\n\n'+report
    card+='\n\n문서 업데이트는 가중치를 바꾸지 않습니다. 실제 GPU 재로딩 검증 리비전은 `'+proof['commit']+'`입니다.\n'
    operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode()))
    operations.append(CommitOperationAdd(path_in_repo='operator-report-20260909/manifest.json',
        path_or_fileobj=json.dumps(manifest,ensure_ascii=False,indent=2).encode()))
    api=HfApi();assert api.model_info(CONFIG['output_repo']).private
    result=api.create_commit(CONFIG['output_repo'],operations=operations,
        commit_message='Document RL pilot limitations, full evaluation, deployment and final resource audit')
    write('report_hub_commit.json',{'time':time.time(),'repo':CONFIG['output_repo'],
        'documentation_commit':result.oid,'gpu_validated_model_commit':proof['commit'],
        'files':len(operations),'weights_modified':False})
    print(json.dumps({'documentation_commit':result.oid,'files':len(operations)}))


if __name__=='__main__':run()
