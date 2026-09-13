"""Exercise all three features through the same callbacks as the local UI."""
import json
import time

from gradio_client import Client, handle_file

from .core import OUT, write
from serving.rl_client import policies, RELEASE, resolve_response


def run():
    deployed = json.loads((OUT/'serving/deployed.json').read_text())
    assert deployed['config']['env']['CLAIM_ADAPTER_REPO'] == 'Mepeng22/gemma-4-31b-claim-rl-v3'
    assert deployed['config']['env']['ENABLE_LORA'] == 'true'
    assert (OUT/'termination.json').exists()
    samples = policies()
    drawing = samples['claimset-US20170065978A1_claim1']
    annotation_id = 'claimset-US20040184954A1_claim37'
    advisory_id = 'KR-2005HEO7354-A1'
    client = Client('http://127.0.0.1:7860',verbose=False)
    # Queue together so max-one worker can process consecutively within idle=5s.
    tasks = [
        ('drawing', drawing['episode_id'], client.submit(
            [handle_file(str(RELEASE/f)) for f in drawing['images']], '', 'ko',
            'claim-v3', False, False, 0., 1024, api_name='/generate_claim')),
        ('annotation', annotation_id, client.submit(annotation_id, 1536,
            '각 종속항 text에서 인용하는 항 번호를 제1항과 같이 명시하세요. '
            '제공된 판례 카드 중 청구항 범위 작성에 실제로 관련 있는 카드에 대해 annotations를 작성하세요. '
            'claim_number, card_id, mode="provided_during_drafting", application을 포함하세요. '
            'application에는 해당 판례 법리와 이 청구항에 적용한 범위·한계를 구체적으로 적으세요. '
            '판례에서 도면에 없는 기술 구성을 가져오지 마세요.', api_name='/test_rl_example')),
        ('advisory', advisory_id, client.submit(advisory_id, 1024,
            '제공된 한국 판례와 가상 사안을 기준으로 결론, 가장 강한 반대 논거, '
            '추가 확인이 필요한 사실을 구분해 답하세요. 현행법 전체를 확인했다고 단정하지 마세요.',
            api_name='/test_rl_example')),
    ]
    results = []
    for feature, eid, job in tasks:
        value = None
        try:
            value = job.result(timeout=760)
            raw = value[0]
            parsed, sources = resolve_response(raw,samples[eid],allow_code_fence=True)
            metadata = value[-1]
            if isinstance(metadata,list): metadata = metadata[0]
            assert metadata['model'] == 'claim-v3'
            assert metadata['finish_reason'] != 'length'
            if feature == 'drawing':
                assert len(parsed['claims']) >= 2
            elif feature == 'annotation':
                assert parsed['annotations'] and sources, 'no_actual_annotations'
            else:
                assert parsed['citations'] and sources, 'no_actual_citations'
            result = {'feature':feature,'episode_id':eid,'passed':True,'raw':raw,
                'parsed':parsed,'sources':sources,'metadata':metadata,'ui_result':value}
        except Exception as exc:
            result = {'feature':feature,'episode_id':eid,'passed':False,
                      'error_type':type(exc).__name__}
            if value is not None: result['ui_result'] = value
        results.append(result)
        write('serving/ui_smoke.json',{'time':time.time(),'complete':len(results)==3,
            'passed':len(results)==3 and all(x['passed'] for x in results),'results':results})
        print(feature,result['passed'],flush=True)
    return results


if __name__=='__main__':run()
