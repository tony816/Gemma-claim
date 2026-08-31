"""청구항 작성 웹 UI — RunPod Serverless(Gemma 4 31B) 앞단.

Hugging Face Space에서 돌아갑니다. RunPod API 키는 Space Secret으로만 들어오고
브라우저로 내려가지 않습니다. 이 Space는 반드시 비공개로 두세요. 공개하면
접근한 사람이 소유자의 RunPod GPU를 쓰게 됩니다.

Secret (Space 설정 → Variables and secrets):
    RUNPOD_API_KEY       필수
    RUNPOD_ENDPOINT_ID   선택, 생략하면 아래 기본값
"""

from __future__ import annotations

import base64
import inspect
import mimetypes
import os
import time

import gradio as gr

from claim_prompt import SYSTEM_PROMPT, sanitise

ENDPOINT = os.environ.get("RUNPOD_ENDPOINT_ID", "fdiltabt78bogm").strip()
API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
MODEL = "gemma4-31b"

# AMPERE_48(A40) 단일 풀, 유휴 300초. 엔드포인트 설정과 맞춰 둡니다.
USD_PER_HOUR = 1.22
IDLE_TIMEOUT = 300
USD_TO_KRW = 1380

# 컨텍스트 16384 토큰. 도면 1장이 약 256 토큰이고 히스토리와 출력이 함께 들어갑니다.
MAX_HISTORY_TURNS = 12
MAX_IMAGE_MB = 8

# Gradio 5는 Chatbot에 type="messages"를 넘겨야 dict 형식을 쓰고, 6은 그것이
# 기본이라 인자 자체가 없습니다. Space가 어느 쪽을 설치할지 고정하지 않았으므로
# 시그니처를 보고 맞춥니다.
_CHATBOT_KW = (
    {"type": "messages"}
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters
    else {}
)
# 같은 이유로 show_copy_button은 5에만 있습니다.
_COPY_KW = (
    {"show_copy_button": True}
    if "show_copy_button" in inspect.signature(gr.Textbox.__init__).parameters
    else {}
)

COLD_START_NOTE = (
    "첫 질문은 모델을 올리느라 5분 정도 걸립니다. 그 뒤로는 5~7초입니다. "
    "마지막 질문 후 5분이 지나면 워커가 잠들고 과금이 멈춥니다."
)


def _client():
    """요청 시점에 만듭니다 — 키가 없을 때 앱이 뜨긴 해야 안내를 보여줄 수 있습니다."""
    if not API_KEY:
        raise gr.Error(
            "RUNPOD_API_KEY가 설정되지 않았습니다. "
            "Space 설정 → Variables and secrets 에서 Secret으로 추가하세요."
        )
    from openai import OpenAI

    return OpenAI(
        api_key=API_KEY,
        base_url=f"https://api.runpod.ai/v2/{ENDPOINT}/openai/v1",
        timeout=900.0,  # 콜드 스타트를 기다려야 합니다.
    )


def encode_image(path: str) -> str:
    mb = os.path.getsize(path) / 1024 / 1024
    if mb > MAX_IMAGE_MB:
        raise gr.Error(f"이미지가 너무 큽니다: {os.path.basename(path)} ({mb:.1f}MB, 최대 {MAX_IMAGE_MB}MB)")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def cost_line(meter: dict) -> str:
    first, last = meter.get("first"), meter.get("last")
    if first is None:
        return "아직 호출 없음 · $0"
    if last is None:
        # 첫 호출이 아직 안 끝났습니다. 끝나야 가동 구간이 정해집니다.
        return "첫 응답 대기 중 · 모델 적재에 약 5분"
    seconds = (last - first) + IDLE_TIMEOUT
    usd = seconds / 3600 * USD_PER_HOUR
    return (f"{meter['turns']}턴 · GPU 가동 약 {seconds / 60:.0f}분"
            f"(유휴 {IDLE_TIMEOUT // 60}분 포함) · ${usd:.3f} (약 {usd * USD_TO_KRW:,.0f}원)")


def respond(text, files, history, meter, claim_mode, temperature, max_tokens):
    """스트리밍 응답. history는 OpenAI 형식, display는 Chatbot 형식."""
    text = (text or "").strip()
    files = files or []

    if not text and not files:
        yield history, meter, _display(history), "", cost_line(meter), gr.update()
        return

    if not text:
        text = ("이 도면들에 나타난 발명에 대한 독립항을 작성해줘."
                if claim_mode else "이 이미지를 설명해줘.")

    # 도면은 업로드된 순서를 그대로 유지합니다 — figure 순서가 의미를 가집니다.
    paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    urls = [encode_image(p) for p in paths]

    if urls:
        parts = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
        parts.append({"type": "text", "text": text})
        history = history + [{"role": "user", "content": parts}]
    else:
        history = history + [{"role": "user", "content": text}]

    label = text if not urls else f"{text}\n\n（도면 {len(urls)}장: {', '.join(os.path.basename(p) for p in paths)}）"
    display = _display(history[:-1]) + [{"role": "user", "content": label},
                                        {"role": "assistant", "content": ""}]

    msgs = []
    if claim_mode:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs += history[-MAX_HISTORY_TURNS * 2:]

    started = time.monotonic()
    if meter.get("first") is None:
        meter = dict(meter, first=started)
        display[-1]["content"] = f"_{COLD_START_NOTE}_"
        yield history, meter, display, "", cost_line(meter), gr.update()

    chunks: list[str] = []
    try:
        stream = _client().chat.completions.create(
            model=MODEL, messages=msgs, stream=True,
            temperature=temperature, max_tokens=int(max_tokens),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                chunks.append(piece)
                display[-1]["content"] = "".join(chunks)
                yield history, meter, display, "", cost_line(meter), gr.update()
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001
        history = history[:-1]
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    answer = "".join(chunks)
    # 히스토리에는 모델이 실제로 낸 말을 남깁니다. 정리본은 따로 보여줄 뿐입니다.
    history = history + [{"role": "assistant", "content": answer}]
    meter = dict(meter, last=time.monotonic(), turns=meter.get("turns", 0) + 1)

    cleaned = ""
    if claim_mode and answer.strip():
        text_out, removed = sanitise(answer)
        if removed and text_out and text_out != answer.strip():
            cleaned = f"{text_out}\n\n---\n정리 항목: {', '.join(removed)}"
        else:
            cleaned = text_out

    yield history, meter, _display(history), "", cost_line(meter), gr.update(value=cleaned)


def _display(history: list[dict]) -> list[dict]:
    out = []
    for m in history:
        c = m["content"]
        if isinstance(c, list):
            texts = [p["text"] for p in c if p.get("type") == "text"]
            imgs = sum(1 for p in c if p.get("type") == "image_url")
            c = (texts[0] if texts else "") + (f"\n\n（도면 {imgs}장）" if imgs else "")
        out.append({"role": m["role"], "content": c})
    return out


def reset():
    return [], {"turns": 0}, [], "", cost_line({}), ""


with gr.Blocks(title="청구항 작성", fill_height=True) as demo:
    gr.Markdown("## 청구항 작성 · Gemma 4 31B")
    if not API_KEY:
        gr.Markdown(
            "> **RUNPOD_API_KEY가 없습니다.** Space 설정 → *Variables and secrets* 에서 "
            "`RUNPOD_API_KEY` 를 **Secret** 으로 추가한 뒤 Space를 재시작하세요."
        )
    gr.Markdown(f"_{COLD_START_NOTE}_")

    history = gr.State([])
    meter = gr.State({"turns": 0})

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=460, show_label=False, **_CHATBOT_KW)
            with gr.Row():
                box = gr.Textbox(placeholder="질문을 적거나, 도면만 올리고 전송하세요.",
                                 show_label=False, scale=5, lines=2)
                send = gr.Button("전송", variant="primary", scale=1)
            files = gr.File(label="도면 (여러 장이면 figure 순서대로 올리세요)",
                            file_count="multiple",
                            file_types=["image"], height=110)
        with gr.Column(scale=2):
            claim_mode = gr.Checkbox(value=True, label="청구항 모드",
                                     info="끄면 시스템 프롬프트 없이 일반 대화")
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="temperature")
            max_tokens = gr.Slider(256, 4096, value=2048, step=256, label="최대 답변 길이")
            cleaned = gr.Textbox(label="정리된 청구항", lines=12,
                                 info="모델이 형식을 어겼을 때만 내용이 달라집니다.",
                                 **_COPY_KW)
            cost = gr.Markdown(cost_line({}))
            clear = gr.Button("대화 초기화")

    outs = [history, meter, chatbot, box, cost, cleaned]
    ins = [box, files, history, meter, claim_mode, temperature, max_tokens]
    send.click(respond, ins, outs)
    box.submit(respond, ins, outs)
    clear.click(reset, None, outs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
