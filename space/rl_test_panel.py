"""Local preview of verified private RL examples."""
import json
import copy
import os
from pathlib import Path
import sys

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serving.rl_client import RELEASE, policies, evidence_in_policy, build_messages, resolve_response
from claim_client import DEFAULT_ENDPOINT, TUNED_MODEL, extract_text, iter_job, response_metadata


def request_policy_for_ui(policy):
    result=copy.deepcopy(policy)
    instruction=('JSON 출력 계약을 유지하세요. 독립항 1개와 근거가 있는 종속항 2개를 작성하고, '
        '각 종속항 text의 첫 부분에 실제로 인용하는 부모 항 번호를 제1항 또는 claim 1처럼 명시하세요. '
        '제공된 판례 카드가 청구항 작성에 관련되면 annotations에 해당 법리의 적용과 한계를 설명하세요. '
        '관련 카드가 없거나 적용하지 않았다면 판례를 만들어 넣지 말고 abstentions에 이유를 적으세요.'
        if policy['task_type']=='claim_set' else
        'JSON 출력 계약을 유지하세요. answer의 첫 문장에서 관할과 제공된 판례 선고 시점 기준의 분석임을 밝히세요. '
        '그 범위에서 적용 법리, 사안에의 적용, 가장 강한 반대 논거와 조건부 결론을 구분하고, '
        'limitations에는 추가로 확인할 사실과 분석의 한계를 적으세요. 현행법을 확인했다고 단정하지 마세요.')
    result['messages'].append({'role':'user','content':instruction})
    return result


def add_panel():
    examples = policies()
    choices = []
    for eid, policy in examples.items():
        cards, _ = evidence_in_policy(policy)
        kind = '판례 자문' if policy['task_type'] == 'patent_advisory' else (
            '청구항 + 판례 어노테이션' if cards else '독립항 + 종속항')
        choices.append((f'{kind} · {eid}', eid))
    first = next(eid for eid, p in examples.items() if evidence_in_policy(p)[0])

    def preview(eid):
        policy = request_policy_for_ui(examples[eid])
        return ('\n\n'.join(m['content'] for m in policy['messages']),
                [str(RELEASE/image) for image in policy['images']])

    def generate_example(eid, max_tokens, custom_request=''):
        if os.environ.get('CLAIM_ENDPOINT_PAUSED') == '1':
            raise gr.Error('서버리스 요청이 중지되어 있습니다. RL_RESULTS.md의 배포 실패 내역을 확인하세요.')
        key = os.environ.get('RUNPOD_API_KEY', '').strip()
        if not key:
            raise gr.Error('프로젝트 .env의 RunPod 인증을 확인하세요.')
        policy = examples[eid]
        request_policy = request_policy_for_ui(policy)
        if custom_request.strip():
            request_policy['messages'].append({'role':'user','content':
                '다음은 이번 테스트의 새 사안 또는 추가 요청입니다. 예제 사실과 다르면 아래 사실을 사용하되, '
                '위에서 제공한 판례 근거·관할·기준 시점·출력 형식을 유지하세요. '
                '청구항 작성에서는 도면에 없는 기술 구성을 추가하지 마세요.\n'+custom_request.strip()})
        yield '', None, None, f'{TUNED_MODEL} 요청 중. 이 호출은 강화학습을 실행하지 않습니다.', None
        try:
            for result in iter_job(os.environ.get('RUNPOD_ENDPOINT_ID', DEFAULT_ENDPOINT), key, build_messages(request_policy),
                                   int(max_tokens), 0., TUNED_MODEL):
                if result.get('status') != 'COMPLETED':
                    yield '', None, None, f"{TUNED_MODEL} · {result.get('status')}", None
                    continue
                metadata = response_metadata(result, TUNED_MODEL)
                raw = extract_text(result)
                try:
                    parsed, sources = resolve_response(raw, policy, allow_code_fence=True)
                    note = f'{TUNED_MODEL} 응답의 형식과 제공된 판례 ID 연결을 확인했습니다. 내용의 법률적 타당성을 검증한 점수는 아닙니다.'
                    if raw.strip().startswith('```'):
                        metadata['display_unwrapped_markdown_fence'] = True
                        note = '표시를 위해 Markdown 코드 블록의 바깥 표시만 제거했습니다. 원문은 그대로 보존합니다. ' + note
                except (ValueError, TypeError, KeyError):
                    parsed, sources = None, None
                    note = f'{TUNED_MODEL} 응답이 출력 계약 또는 판례 연결 검사를 통과하지 못했습니다. 원문을 확인하세요.'
                yield raw, parsed, sources, note, metadata
        except Exception as exc:
            yield '', None, None, f'요청 실패: {type(exc).__name__}. 인증과 엔드포인트 상태를 확인하세요.', None

    with gr.Accordion('독립항·종속항 / 판례 어노테이션 / 판례 자문 · 확정 예제', open=False):
        model_note = ('기존 v2 (`claim-v2`). v3 배포 검증 실패로 복구했으며 현재 요청은 중지되어 있습니다.'
                      if TUNED_MODEL == 'claim-v2' else f'추가 강화학습 모델 `{TUNED_MODEL}`.')
        gr.Markdown(f'현재 연결 모델: **{model_note}** '
                    '확정된 학습 예제로 출력 계약을 시험합니다. 최종 평가 자료와 참조 답안은 화면에 전달하지 않습니다.')
        example = gr.Dropdown(choices, value=first, label='시험할 확정 예제')
        ask, images = preview(first)
        text = gr.Textbox(value=ask, label='모델에 전달되는 사실·지시·판례 근거', lines=8, interactive=False)
        gallery = gr.Gallery(value=images, label='제공 도면 · 원래 순서', columns=4)
        custom_request = gr.Textbox(label='새 가상사안 / 추가 요청 (선택)', lines=3,
            placeholder='비워두면 확정 예제를 사용합니다. 새 사실이나 질문을 입력하면 선택한 예제의 판례 근거를 유지해 답합니다.')
        length = gr.Slider(256, 2048, value=1024, step=256, label='최대 답변 토큰')
        submit = gr.Button(f'{TUNED_MODEL}로 시험 · 요청 과금 발생', interactive=os.environ.get('CLAIM_ENDPOINT_PAUSED') != '1')
        status = gr.Markdown('판례는 제공된 카드와 연결하며, 모델 내부의 인과적 출처를 뜻하지 않습니다.')
        raw = gr.Textbox(label='모델 원문', lines=10, interactive=False)
        parsed = gr.JSON(label='출력 계약 확인 결과')
        sources = gr.JSON(label='제공 카드에서 확인한 사건번호·법원·날짜·페이지·원문 해시')
        metadata = gr.JSON(label='실제 응답 모델·실행 정보')
        example.change(preview, example, [text, gallery], queue=False)
        submit.click(generate_example, [example, length, custom_request], [raw, parsed, sources, status, metadata],
                     concurrency_limit=1, concurrency_id='gpu', api_name='test_rl_example')
