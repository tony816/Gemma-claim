"""Operational redeployment checks after the user accepted known pilot limitations.

This never changes or relabels the earlier quality evaluation.
"""
import json
import time

from gradio_client import Client, handle_file
from tools.rl_cloud import runpod
from serving.rl_client import policies, RELEASE
from .core import OUT, CONFIG, sha, write
from .serve import snapshot, private_template


def smoke():
    deployed=json.loads((OUT/'serving/deployed.json').read_text())
    assert deployed['config']['env']['CLAIM_ADAPTER_REVISION']=='e5aac01e9fe0af54b17e7f52b66486112183035e'
    example=policies()['claimset-US20170065978A1_claim1']
    drawing=RELEASE/example['images'][0]
    prompt='도면에 보이는 형태만 한국어 한 문장(30자 이내)으로 설명하세요. 특허 청구항이나 JSON을 쓰지 마세요.'
    client=Client('http://127.0.0.1:7860',verbose=False)
    started=time.time()
    job=client.submit([handle_file(str(drawing))],prompt,'ko','claim-v3',False,False,0.,256,
                      api_name='/generate_claim')
    result=job.result(timeout=800)
    metadata=result[-1]
    assert len(metadata)==1 and metadata[0]['model']=='claim-v3','actual_response_model_mismatch'
    assert result[0].strip(),'empty_inference'
    write('serving/operational_smoke.json',{'time':time.time(),'started':started,'passed':True,
        'scope':'One short actual image request through local Gradio; operational alias/routing only, not claim/annotation/legal quality acceptance.',
        'prior_quality_failures_unchanged':True,'prompt':prompt,'image':str(drawing.relative_to(RELEASE)),
        'image_sha256':sha(drawing),'raw':result[0],'metadata':metadata[0],'ui_result':result})
    print(json.dumps({'passed':True,'metadata':metadata[0],'raw':result[0]},ensure_ascii=False))


def disarm():
    proof=json.loads((OUT/'serving/operational_smoke.json').read_text())
    assert proof['passed'] and proof['metadata']['model']=='claim-v3'
    text=(OUT/'serving/worker_logs.jsonl').read_text()
    assert 'CLAIM_PINNED_ADAPTER_VERIFIED' in text
    assert CONFIG['resume_from']['adapter_sha256'] in text
    assert 'e5aac01e9fe0af54b17e7f52b66486112183035e' in text
    state=snapshot()
    assert not state['active_pods'] and state['account']['currentSpendPerHr']==0
    assert state['config']['workers']=={'min':0,'max':1,'idleTimeout':5}
    template=json.loads((OUT/'serving/template_created.json').read_text())
    private_template(template['id'])
    current=runpod('/templates/'+template['id'],v1=True)
    env=dict(current['env']);env.pop('RUNPOD_API_KEY',None);env.pop('CLAIM_DEPLOYMENT_DEADLINE',None)
    runpod('/templates/'+template['id'],'PATCH',{'env':env},v1=True)
    current=runpod('/serverless/fdiltabt78bogm')
    assert 'RUNPOD_API_KEY' not in current['env'] and 'CLAIM_DEPLOYMENT_DEADLINE' not in current['env']
    write('serving/guard_disarmed.json',{'time':time.time(),'operational_verification_passed':True,
        'quality_evaluation_changed':False,'control_key_removed':True,'template_id':template['id']})


if __name__=='__main__':
    import sys
    if sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='disarm':disarm()
