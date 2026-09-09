"""Private drawing and case tester with verified response-model routing."""
from __future__ import annotations

import json
import os
import time
import urllib.error

import gradio as gr

from claim_client import (
    DEFAULT_ENDPOINT, SERVED_MODEL, TUNED_MODEL, TRAINING_PROMPTS,
    build_content, extract_text, iter_job, response_metadata,
)
from claim_prompt import sanitise, system_prompt

ENDPOINT = os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT).strip()
API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
IS_RL = TUNED_MODEL == 'claim-v3'
IS_PAUSED = os.environ.get('CLAIM_ENDPOINT_PAUSED') == '1'
MODEL_CHOICES = ([("추가 강화학습 v3", TUNED_MODEL)] if IS_RL else
                 [("파인튜닝 v2", TUNED_MODEL), ("베이스 모델", SERVED_MODEL)])


def model_defaults(model):
    return model == SERVED_MODEL


def generate(files, prompt, lang, model, use_system, compare, temperature, max_tokens):
    if IS_PAUSED:
        raise gr.Error('CLAIM_ENDPOINT_PAUSED=1 설정으로 요청이 중지되어 있습니다.')
    if not API_KEY:
        raise gr.Error("프로젝트 .env에 RUNPOD_API_KEY를 설정하고 화면을 다시 실행하세요.")
    if not files:
        raise gr.Error("테스트할 도면을 한 장 이상 올려주세요.")
    paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    if len(paths) > 8:
        raise gr.Error("도면은 최대 8장까지 올릴 수 있습니다.")
    if any(os.path.getsize(p) > 8 * 1024 * 1024 for p in paths):
        raise gr.Error("도면 한 장의 크기는 최대 8MB입니다.")
    content = build_content(paths, None, lang)
    if (prompt or "").strip():
        content[-1]["text"] = prompt.strip()

    raw = clean = other_raw = other_clean = ""
    metadata = []
    targets = [(model, use_system)]
    if compare and not IS_RL:
        other = SERVED_MODEL if model == TUNED_MODEL else TUNED_MODEL
        targets.append((other, other == SERVED_MODEL))
    started = time.monotonic()
    for index, (target, with_system) in enumerate(targets):
        messages = ([{"role": "system", "content": system_prompt(lang)}] if with_system else [])
        messages.append({"role": "user", "content": content})
        yield raw, clean, other_raw, other_clean, f"{target} 요청 중 · 워커 배정과 첫 로딩을 기다립니다.", metadata
        try:
            for result in iter_job(ENDPOINT, API_KEY, messages, int(max_tokens), float(temperature), target):
                if result.get("status") != "COMPLETED":
                    elapsed = int(time.monotonic() - started)
                    yield raw, clean, other_raw, other_clean, f"{target} · {result.get('status')} · {elapsed}초 경과", metadata
                    continue
                info = response_metadata(result, target)
                answer = extract_text(result)
                if not answer or not answer.strip():
                    raise RuntimeError("Empty completion")
                if IS_RL:
                    cleaned, removed = answer, []
                    candidate = answer.strip()
                    if candidate.startswith('```json') and candidate.endswith('```'):
                        candidate = candidate[7:-3].strip()
                    try:
                        cleaned = json.dumps(json.loads(candidate), ensure_ascii=False, indent=2)
                    except ValueError:
                        info['json_valid'] = False
                    else:
                        info['json_valid'] = True
                else:
                    cleaned, removed = sanitise(answer, lang)
                info.update(system_prompt=with_system, sanitiser_removed=removed)
                metadata.append(info)
                if index == 0:
                    raw, clean = answer, cleaned
                else:
                    other_raw, other_clean = answer, cleaned
        except Exception as exc:
            # Never display HTTP bodies, request payloads, or credentials.
            if isinstance(exc, urllib.error.HTTPError):
                detail = f"HTTP {exc.code}"
            elif isinstance(exc, TimeoutError):
                detail = "응답 대기 한도 초과. 요청 취소를 시도했습니다."
            else:
                detail = type(exc).__name__
            note = f"{target} 호출 실패 ({detail}). RunPod의 모델 설정과 워커 로그를 확인하세요."
            yield raw, clean, other_raw, other_clean, note, metadata
            return
    truncated = any(m.get("finish_reason") == "length" for m in metadata)
    note = "완료 · 응답의 모델 ID가 선택한 모델과 일치함을 확인했습니다."
    if truncated:
        note += " 최대 답변 길이에 도달해 잘린 결과가 있습니다."
    yield raw, clean, other_raw, other_clean, note, metadata


with gr.Blocks(title="Gemma 청구항 · 판례 테스트") as demo:
    gr.Markdown('## 요청 중지\n로컬 설정 CLAIM_ENDPOINT_PAUSED=1이 모델 요청을 막고 있습니다.', visible=IS_PAUSED)
    gr.Markdown("# 도면으로 독립항·종속항 작성" if IS_RL else "# 도면으로 청구항 작성")
    gr.Markdown("기존 v2에서 추가 강화학습한 `claim-v3`에 연결됩니다. 비공개 확정 예제 자료가 설치된 환경에서는 판례 어노테이션과 자문 패널도 표시됩니다." if IS_RL else
                ("추가 강화학습과 저장은 끝났고, 배포 검증 실패로 v2 복구 후 요청을 중지했습니다." if IS_PAUSED else "기존 파인튜닝 v2 모델에 연결됩니다."))
    gr.Markdown("매번 독립된 요청으로 테스트합니다. 판례 카드는 제공된 근거이며 모델 내부의 인과적 출처를 뜻하지 않습니다.")
    gr.Markdown("파일럿 평가에서 도면 근거, 종속항 인용과 판례 어노테이션 오류가 남아 있습니다. 생성된 원문과 제공 근거를 함께 확인하세요.", visible=IS_RL)
    with gr.Row():
        with gr.Column(scale=2):
            files = gr.File(label="발명 도면 · 그림 순서대로, 최대 8장", file_count="multiple", file_types=["image"], allow_reordering=True)
            prompt = gr.Textbox(label="요청 문구 (선택)", placeholder="비워두면 도면에 근거한 독립항·종속항 작성 요청을 사용합니다." if IS_RL else "비워두면 v2 학습 요청 문구를 사용합니다.", lines=2)
            with gr.Row():
                model = gr.Radio(MODEL_CHOICES, value=TUNED_MODEL, label="테스트 모델")
                lang = gr.Radio([("한국어", "ko"), ("English", "en")], value="ko", label="언어")
            use_system = gr.Checkbox(value=False, label="청구항 작성용 시스템 프롬프트 추가", visible=not IS_RL)
            compare = gr.Checkbox(value=not IS_RL, label="같은 도면으로 다른 모델도 비교", visible=not IS_RL, info="켜면 모델별로 한 번씩, 총 2회 호출합니다.")
            with gr.Accordion("생성 설정", open=False):
                temperature = gr.Slider(0, 1, value=0, step=0.05, label="Temperature")
                max_tokens = gr.Slider(256, 2048, value=1024, step=256, label="최대 답변 토큰")
                gr.Markdown("비교 모델은 기본 설정(v2: 시스템 프롬프트 없음, 베이스: 있음)을 사용합니다.", visible=not IS_RL)
            submit = gr.Button("청구항 생성 · 요청 과금 발생", variant="primary", interactive=not IS_PAUSED)
            gr.Markdown("RunPod는 워커 기동·처리·종료 전 대기 동안 과금됩니다. 설정한 유휴 대기는 5초이며 실제 종료는 더 걸릴 수 있습니다. 실제 단가와 청구액은 RunPod에서 확인하세요.")
        with gr.Column(scale=3):
            status = gr.Markdown("요청 중지 상태입니다. 확정 예제와 근거는 아래에서 열람할 수 있습니다." if IS_PAUSED else "도면을 올린 뒤 생성 버튼을 누르세요.")
            with gr.Row():
                raw = gr.Textbox(label="선택한 모델 · 원문", lines=15, interactive=False, buttons=["copy"])
                other_raw = gr.Textbox(label="비교 모델 · 원문", lines=15, interactive=False, buttons=["copy"], visible=not IS_RL)
            with gr.Accordion("정리된 청구항 (원문과 별도)", open=False):
                clean = gr.Textbox(label="선택한 모델 · 정리본", lines=8, interactive=False, buttons=["copy"])
                other_clean = gr.Textbox(label="비교 모델 · 정리본", lines=8, interactive=False, buttons=["copy"], visible=not IS_RL)
            with gr.Accordion("실제 응답 모델 및 실행 정보", open=False):
                metadata = gr.JSON(label="응답 정보")
    model.change(model_defaults, model, use_system, queue=False)
    submit.click(generate, [files, prompt, lang, model, use_system, compare, temperature, max_tokens],
                 [raw, clean, other_raw, other_clean, status, metadata], concurrency_limit=1,
                 concurrency_id="gpu", api_name="generate_claim")


_project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_has_materials = os.path.isfile(os.path.join(_project, 'data', 'case_rl', 'releases',
    'rl-pilot-27fc35d39288134b', 'manifest.json')) and os.path.isfile(
    os.path.join(_project, '.superloopy', 'evidence', 'frozen-release.json'))
if _has_materials:
    from rl_test_panel import add_panel
    with demo:
        add_panel()
else:
    with demo:
        gr.Markdown('판례 확정 예제 패널: 비공개 자료와 검증 영수증을 설치하면 표시됩니다. 설치 방법은 serving/README.md를 확인하세요. 도면 생성은 해당 자료 없이 사용할 수 있습니다.')


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1', server_port=7860, share=False)
