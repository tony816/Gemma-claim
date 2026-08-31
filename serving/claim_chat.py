#!/usr/bin/env python3
"""
청구항 작성 대화창  ·  RunPod / DeepInfra

tony816/Gemma 의 gemma_chat.py 를 Gemma-claim 용으로 맞춘 것입니다.
기본 시스템 프롬프트가 청구항 작성용으로 들어가 있고, /image 가 도면을
여러 장 순서대로 받습니다. 일반 대화로 쓰려면 --plain 을 주세요.

터미널에서 주고받는 대화형 클라이언트.
답변은 한 글자씩 흘러나오고, 이전 대화를 기억하며, 비용을 계속 누적해 보여줍니다.

준비 (최초 1회):
    pip install openai
    pip install pillow      # 클립보드 이미지 첨부를 쓸 때만

    # RunPod 사용 시 (기본값)
    export RUNPOD_API_KEY='런팟_키'
    export RUNPOD_ENDPOINT_ID='fdiltabt78bogm'   # 생략하면 이 값이 기본

    # DeepInfra 사용 시
    export DEEPINFRA_TOKEN='발급받은_키'

실행:
    python gemma_chat.py
    python gemma_chat.py --provider deepinfra
    python gemma_chat.py --system "당신은 한국 특허 실무 전문가입니다."
    python gemma_chat.py --load 어제대화.json

RunPod은 GPU를 시간당 빌리는 방식이라 토큰이 아니라 '깨어 있는 시간'에
비례해 돈이 나갑니다. 첫 질문은 모델을 올리느라 5~6분 걸리고, 그 뒤로는
빠릅니다. 마지막 질문 후 10분이 지나면 워커가 잠들고 과금도 멈춥니다.

대화 중 명령어:
    /help          명령어 목록
    /image <경로...>  도면 첨부. 여러 장이면 figure 순서대로 나열
    /image         경로 없이 쓰면 클립보드의 이미지를 첨부
    /clip, /v      클립보드의 이미지를 첨부
    (경로만 입력)  터미널에 도면을 끌어다 놓으면 그대로 첨부
    /paste         여러 줄 붙여넣기 (긴 청구항 등)
    /new           대화 초기화
    /undo          마지막 주고받기 취소
    /save [파일]   대화 저장
    /load <파일>   대화 불러오기
    /lang en|ko    청구항 언어 바꾸기
    /system <문장> 시스템 프롬프트 변경
    /cost          누적 비용
    /exit          종료 (Ctrl+D 도 동일)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

try:  # 방향키로 이전 입력 불러오기. 없으면 그냥 넘어갑니다.
    import readline  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from claim_prompt import (SYSTEM_PROMPT as CLAIM_SYSTEM,
                              sanitise as claim_sanitise,
                              system_prompt as claim_system_prompt)
except ImportError:  # 단독으로 복사해 쓸 때
    CLAIM_SYSTEM, claim_sanitise, claim_system_prompt = None, None, None

USD_TO_KRW = 1380

# 이 프로젝트의 엔드포인트(fdiltabt78bogm)는 AMPERE_48(A40) 단일 풀입니다.
RUNPOD_USD_PER_HOUR = 1.22
RUNPOD_IDLE_TIMEOUT = 300   # 엔드포인트에 설정된 값. 마지막 질문 후 이만큼 더 과금됨.
DEFAULT_ENDPOINT = "fdiltabt78bogm"
# 엔드포인트 컨텍스트는 16384 토큰입니다. 도면 1장이 약 256 토큰이고,
# 히스토리와 출력이 그 안에 함께 들어가야 합니다.
MAX_MODEL_LEN = 16384

PROVIDERS = {
    "runpod": {
        "label": "RunPod",
        "model": "gemma4-31b",
        "key_env": "RUNPOD_API_KEY",
        "billing": "time",
    },
    "deepinfra": {
        "label": "DeepInfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "google/gemma-4-31B-it",
        "key_env": "DEEPINFRA_TOKEN",
        "billing": "token",
        "price_in": 0.104,    # flex 요금제, USD per 1M tokens
        "price_out": 0.304,
    },
}

# 히스토리가 길어지면 오래된 것부터 버립니다 (모델은 262K까지 받지만 비용 관리용)
MAX_HISTORY_TURNS = 12

# Gemma 4는 멀티모달이라 이미지도 받습니다. 도면·그래프·스크린샷 등.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MAX_IMAGE_MB = 8

C_DIM = "\033[2m"
C_USER = "\033[1;36m"
C_BOT = "\033[1;32m"
C_WARN = "\033[1;33m"
C_OFF = "\033[0m"

if os.name == "nt" and not os.environ.get("WT_SESSION"):
    C_DIM = C_USER = C_BOT = C_WARN = C_OFF = ""

# 한국어 콘솔은 출력 인코딩이 cp949 라, 표현 못 하는 글자 하나에 프로그램이
# 통째로 죽습니다. 인코딩은 콘솔 것을 그대로 두고 실패만 흘려보냅니다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass


class Chat:
    def __init__(self, client, system: str | None, provider: dict,
                 temperature: float = 0.7, max_tokens: int = 2048,
                 postprocess=None):
        self.client = client
        self.system = system
        self.provider = provider
        self.model = provider["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 청구항 모드에서 답변을 다듬어 함께 보여주기 위한 훅.
        self.postprocess = postprocess
        self.history: list[dict] = []
        self.tok_in = 0
        self.tok_out = 0
        self.turns = 0
        # RunPod은 시간 과금이라 워커가 깨어 있던 구간을 추적합니다.
        self.first_call: float | None = None
        self.last_call: float | None = None

    # ---------- 비용 ----------
    @property
    def billed_seconds(self) -> float:
        """워커가 깨어 있었던 것으로 보이는 시간 + 잠들기까지의 유휴 시간."""
        if self.first_call is None or self.last_call is None:
            return 0.0
        return (self.last_call - self.first_call) + RUNPOD_IDLE_TIMEOUT

    @property
    def usd(self) -> float:
        if self.provider["billing"] == "time":
            return self.billed_seconds / 3600 * RUNPOD_USD_PER_HOUR
        return (self.tok_in / 1e6 * self.provider["price_in"]
                + self.tok_out / 1e6 * self.provider["price_out"])

    def cost_line(self) -> str:
        won = f"약 {self.usd * USD_TO_KRW:,.0f}원"
        if self.provider["billing"] == "time":
            mins = self.billed_seconds / 60
            return (f"{C_DIM}누적 {self.turns}턴 · GPU 가동 약 {mins:.0f}분"
                    f"(유휴 {RUNPOD_IDLE_TIMEOUT // 60}분 포함) · ${self.usd:.3f} ({won}){C_OFF}")
        return (f"{C_DIM}누적 {self.turns}턴 · 입력 {self.tok_in:,} / 출력 {self.tok_out:,} tok "
                f"· ${self.usd:.4f} ({won}){C_OFF}")

    # ---------- 메시지 ----------
    def messages(self) -> list[dict]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        trimmed = self.history[-MAX_HISTORY_TURNS * 2:]
        if len(trimmed) < len(self.history):
            print(f"{C_DIM}  (오래된 대화 일부는 비용 절약을 위해 생략됩니다){C_OFF}")
        return msgs + trimmed

    def ask(self, text: str, images: list[str] | None = None) -> None:
        if images:
            parts: list[dict] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
            parts.append({"type": "text", "text": text})
            self.history.append({"role": "user", "content": parts})
        else:
            self.history.append({"role": "user", "content": text})
        started = time.monotonic()
        if self.first_call is None:
            self.first_call = started
            if self.provider["billing"] == "time":
                print(f"{C_DIM}  (첫 질문은 모델을 올리느라 5~6분 걸립니다. 이후는 빠릅니다.){C_OFF}")
        chunks: list[str] = []
        usage = None

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages(),
                stream=True,
                stream_options={"include_usage": True},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            print(f"\n{C_BOT}Gemma{C_OFF}  ", end="", flush=True)
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    chunks.append(piece)
                    print(piece, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print(f"\n{C_WARN}  (중단됨){C_OFF}")
            if not chunks:
                self.history.pop()
                return
        except Exception as exc:  # noqa: BLE001
            print(f"\n{C_WARN}  오류: {type(exc).__name__}: {exc}{C_OFF}")
            self.history.pop()
            return

        answer = "".join(chunks)

        # 모델이 형식을 어기면 다듬은 결과를 아래에 덧붙여 보여줍니다. 히스토리에는
        # 모델이 실제로 낸 말을 그대로 남겨, 다음 턴의 맥락이 어긋나지 않게 합니다.
        if self.postprocess:
            cleaned, removed = self.postprocess(answer)
            if removed and cleaned and cleaned != answer.strip():
                print(f"\n{C_DIM}  ── 정리됨 ({', '.join(removed)}) ──{C_OFF}")
                print(f"{C_BOT}{cleaned}{C_OFF}")

        self.history.append({"role": "assistant", "content": answer})
        self.turns += 1

        if usage:
            self.tok_in += getattr(usage, "prompt_tokens", 0) or 0
            self.tok_out += getattr(usage, "completion_tokens", 0) or 0
        else:  # 서버가 usage를 안 주면 대략 추정 (글자수/2.5)
            self.tok_in += int(sum(
                len(m["content"]) if isinstance(m["content"], str) else 800
                for m in self.messages()
            ) / 2.5)
            self.tok_out += int(len(answer) / 2.5)

        self.last_call = time.monotonic()
        elapsed = self.last_call - started
        print(f"{C_DIM}  {elapsed:.1f}초 · {self.cost_line()[len(C_DIM):]}")

    # ---------- 저장/불러오기 ----------
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "system": self.system,
            "history": self.history,
            "tok_in": self.tok_in,
            "tok_out": self.tok_out,
            "turns": self.turns,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.system = data.get("system") or self.system
        self.history = data.get("history", [])
        self.tok_in = data.get("tok_in", 0)
        self.tok_out = data.get("tok_out", 0)
        self.turns = data.get("turns", 0)


HELP = """
  /image <경로> [질문]   이미지 첨부해서 질문 (도면·그래프·스크린샷)
  /image          경로 없이 쓰면 클립보드의 이미지를 첨부
  /clip [질문]    클립보드의 이미지를 첨부 (/v 도 같음)
  (경로만 입력)   터미널에 파일을 끌어다 놓으면 그대로 첨부
  /paste          여러 줄 붙여넣기 — 다 붙인 뒤 마지막 줄에 . 만 입력
  /new            대화 초기화 (비용 누적도 리셋)
  /undo           마지막 질문과 답변 취소
  /save [파일]    대화 저장 (기본: chat_날짜시각.json)
  /load <파일>    저장한 대화 이어가기
  /lang en|ko     청구항 언어 바꾸기
  /system <문장>  시스템 프롬프트 교체
  /system         현재 시스템 프롬프트 보기
  /cost           누적 비용
  /exit           종료
"""


def load_image(path_str: str) -> str | None:
    """로컬 이미지를 data URI 로. 실패하면 None."""
    path = Path(path_str.strip().strip('"').strip("'")).expanduser()
    if not path.exists():
        print(f"{C_WARN}  파일이 없습니다: {path}{C_OFF}")
        return None
    if path.suffix.lower() not in IMAGE_EXTS:
        print(f"{C_WARN}  지원하지 않는 형식입니다: {path.suffix} "
              f"({', '.join(sorted(IMAGE_EXTS))}){C_OFF}")
        return None
    mb = path.stat().st_size / 1024 / 1024
    if mb > MAX_IMAGE_MB:
        print(f"{C_WARN}  이미지가 너무 큽니다: {mb:.1f}MB (최대 {MAX_IMAGE_MB}MB){C_OFF}")
        return None
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    print(f"{C_DIM}  첨부: {path.name} ({mb:.1f}MB){C_OFF}")
    return f"data:{mime};base64,{b64}"


def strip_quotes(tok: str) -> str:
    """드롭한 경로를 감싼 따옴표를 걷어냅니다."""
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        tok = tok[1:-1]
    return tok


def split_tokens(arg: str) -> list[str]:
    """터미널이 떨어뜨린 줄을 토큰으로 가릅니다.

    드래그앤드롭 형식이 플랫폼마다 다릅니다. Windows Terminal 은 공백이 든
    경로를 따옴표로 감싸고, macOS·Linux 터미널은 공백을 역슬래시로 이스케이프
    합니다. posix 모드를 플랫폼에 맞춰야 둘 다 제대로 풀립니다 — Windows 에서
    posix=True 로 파싱하면 경로 구분자인 역슬래시를 이스케이프로 먹어버립니다.
    """
    posix = os.name != "nt"
    try:
        tokens = shlex.split(arg, posix=posix)
    except ValueError:
        tokens = arg.split()
    return [strip_quotes(t) for t in tokens if t.strip()]


def dnd_paths(raw: str) -> list[str]:
    """줄 전체가 이미지 경로뿐이면 그 경로들을, 아니면 빈 목록을 돌려줍니다.

    터미널에 도면을 끌어다 놓으면 경로만 적힌 줄이 됩니다. /image 를 앞에
    붙이지 않아도 첨부로 받기 위한 판별입니다. 한 토큰이라도 실제 이미지
    파일이 아니면 일반 질문으로 넘깁니다.
    """
    def is_image(tok: str) -> bool:
        path = Path(tok).expanduser()
        return path.suffix.lower() in IMAGE_EXTS and path.exists()

    tokens = split_tokens(raw)
    if tokens and all(is_image(t) for t in tokens):
        return tokens

    # 공백이 든 경로를 따옴표 없이 떨어뜨리는 터미널도 있습니다. 토큰으로
    # 갈랐을 때 안 맞으면 줄 전체를 경로 하나로 보고 다시 봅니다.
    whole = strip_quotes(raw)
    return [whole] if is_image(whole) else []


def clipboard_images() -> list[str]:
    """클립보드에 든 이미지를 data URI 목록으로. 없으면 빈 목록.

    터미널은 이미지 붙여넣기를 글자로 받지 못하므로, 붙여넣는 대신 이 함수가
    클립보드를 직접 읽습니다. 캡처한 비트맵과 탐색기에서 복사한 파일 둘 다
    들어올 수 있어 갈라서 처리합니다.
    """
    try:
        from PIL import ImageGrab
    except ImportError:
        print(f"{C_WARN}  클립보드를 읽으려면 pillow 가 필요합니다.  pip install pillow{C_OFF}")
        return []

    try:
        grabbed = ImageGrab.grabclipboard()
    except Exception as exc:  # Linux 는 xclip/wl-paste 가 있어야 합니다.
        print(f"{C_WARN}  클립보드를 읽지 못했습니다: {exc}{C_OFF}")
        return []

    if grabbed is None:
        print(f"{C_WARN}  클립보드에 이미지가 없습니다. 캡처하거나 파일을 복사한 뒤 다시 시도하세요.{C_OFF}")
        return []

    if isinstance(grabbed, list):  # 탐색기에서 복사한 파일 — 경로로 옵니다.
        return [u for u in (load_image(p) for p in grabbed) if u]

    # 캡처 비트맵. Windows 클립보드는 알파가 0으로 채워져 오는 경우가 있어
    # 알파를 버리고 RGB 로 눕힙니다. 그대로 두면 전부 투명한 그림이 됩니다.
    img = grabbed.convert("RGB") if grabbed.mode not in ("RGB", "L") else grabbed
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    mb = len(data) / 1024 / 1024
    if mb > MAX_IMAGE_MB:
        print(f"{C_WARN}  클립보드 이미지가 너무 큽니다: {mb:.1f}MB (최대 {MAX_IMAGE_MB}MB){C_OFF}")
        return []
    print(f"{C_DIM}  첨부: 클립보드 이미지 {img.width}x{img.height} ({mb:.1f}MB){C_OFF}")
    return ["data:image/png;base64," + base64.b64encode(data).decode()]


def split_image_args(arg: str) -> tuple[list[str], str]:
    """앞쪽에 이어지는 이미지 파일들을 경로로, 나머지를 질문으로 가릅니다.

    도면은 figure 순서가 의미를 가지므로 적힌 순서를 그대로 유지합니다.
    """
    tokens = split_tokens(arg)

    paths: list[str] = []
    rest: list[str] = []
    for i, tok in enumerate(tokens):
        cand = Path(tok).expanduser()
        if cand.suffix.lower() in IMAGE_EXTS and cand.exists():
            paths.append(tok)
        else:
            rest = tokens[i:]
            break

    if paths:
        return paths, " ".join(rest).strip()

    # 따옴표 없이 공백이 든 경로 하나를 적은 경우: 앞에서부터 가장 긴 실제 경로를 찾습니다.
    for sep in range(len(arg), 0, -1):
        cand = arg[:sep].strip().strip('"').strip("'")
        if cand and Path(cand).expanduser().exists():
            return [cand], arg[sep:].strip()
    return [], arg.strip()


def ask_with_images(chat: Chat, urls: list[str], question: str) -> None:
    """첨부한 이미지와 함께 질문합니다. 질문이 비어 있으면 한 번 더 물어봅니다."""
    if not urls:
        return
    if not question:
        try:
            question = input(f"{C_USER}질문{C_OFF}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    default_ask = ("이 도면들에 나타난 발명에 대한 독립항을 작성해줘."
                   if chat.system else "이 이미지를 설명해줘.")
    chat.ask(question or default_ask, images=urls)
    print()


def read_paste() -> str:
    print(f"{C_DIM}  붙여넣고, 마지막 줄에 점(.) 하나만 입력하세요.{C_OFF}")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def handle_command(chat: Chat, raw: str) -> bool:
    """명령어를 처리했으면 True. 아니면 False (= 일반 질문)."""
    cmd, _, arg = raw.partition(" ")
    arg = arg.strip()

    if cmd in ("/exit", "/quit", "/q"):
        raise SystemExit(0)

    if cmd in ("/help", "/?"):
        print(HELP)
    elif cmd == "/new":
        chat.history.clear()
        chat.tok_in = chat.tok_out = chat.turns = 0
        print(f"{C_DIM}  대화를 초기화했습니다. (GPU 가동 시간은 계속 누적됩니다){C_OFF}")
    elif cmd == "/undo":
        if len(chat.history) >= 2:
            chat.history = chat.history[:-2]
            chat.turns = max(0, chat.turns - 1)
            print(f"{C_DIM}  마지막 주고받기를 취소했습니다. (남은 {chat.turns}턴){C_OFF}")
        else:
            print(f"{C_DIM}  취소할 대화가 없습니다.{C_OFF}")
    elif cmd == "/cost":
        print("  " + chat.cost_line())
    elif cmd == "/lang":
        # 청구항 언어를 대화 중에 바꿉니다. 히스토리는 그대로 두므로 앞선
        # 도면과 맥락을 잃지 않고 다음 답만 다른 언어로 받습니다.
        if claim_system_prompt is None:
            print(f"{C_WARN}  claim_prompt.py 를 찾지 못해 쓸 수 없습니다.{C_OFF}")
        elif arg in ("en", "ko"):
            chat.system = claim_system_prompt(arg)
            print(f"{C_DIM}  청구항 언어를 {arg} 로 바꿨습니다.{C_OFF}")
        else:
            print(f"{C_WARN}  /lang en 또는 /lang ko 로 적어주세요.{C_OFF}")
    elif cmd == "/system":
        if arg:
            chat.system = arg
            print(f"{C_DIM}  시스템 프롬프트를 바꿨습니다.{C_OFF}")
        else:
            print(f"{C_DIM}  현재: {chat.system or '(없음)'}{C_OFF}")
    elif cmd == "/save":
        path = Path(arg) if arg else Path(f"chat_{datetime.now():%Y%m%d_%H%M}.json")
        chat.save(path)
        print(f"{C_DIM}  저장: {path.resolve()}{C_OFF}")
    elif cmd == "/load":
        if not arg:
            print(f"{C_WARN}  파일명을 적어주세요. 예: /load 어제대화.json{C_OFF}")
        elif not Path(arg).exists():
            print(f"{C_WARN}  파일이 없습니다: {arg}{C_OFF}")
        else:
            chat.load(Path(arg))
            print(f"{C_DIM}  불러왔습니다. {chat.turns}턴 이어서 진행합니다.{C_OFF}")
    else:
        return False
    return True


def resolve_provider(name: str) -> tuple[dict, str, str]:
    """(provider 설정, base_url, api key) 반환. 준비가 안 됐으면 안내하고 종료."""
    prov = dict(PROVIDERS[name])

    if name == "runpod":
        endpoint = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip() or DEFAULT_ENDPOINT
        base_url = f"https://api.runpod.ai/v2/{endpoint}/openai/v1"
    else:
        base_url = prov["base_url"]

    key = os.environ.get(prov["key_env"], "").strip()
    if not key:
        sys.exit(
            f"{prov['key_env']} 환경변수가 없습니다.\n"
            f"  {prov['label']} 대시보드에서 키를 발급받아 설정하세요.\n"
            f"    export {prov['key_env']}='...'      (macOS/Linux)\n"
            f"    set {prov['key_env']}=...           (Windows)"
        )
    return prov, base_url, key


def main() -> None:
    ap = argparse.ArgumentParser(description="청구항 작성 대화창 (RunPod / DeepInfra)")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="runpod",
                    help="어디로 보낼지 (기본: runpod)")
    ap.add_argument("--system", default=None, help="시스템 프롬프트 (기본: 청구항 작성용)")
    ap.add_argument("--plain", action="store_true",
                    help="청구항 프롬프트 없이 일반 대화로 시작")
    ap.add_argument("--load", default=None, help="저장한 대화 파일로 시작")
    ap.add_argument("--temperature", type=float, default=None,
                    help="기본: 청구항 모드 0.2, 일반 대화 0.7")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="답변 최대 길이 (엔드포인트 컨텍스트 %d 토큰)" % MAX_MODEL_LEN)
    ap.add_argument("--lang", choices=("en", "ko"), default="en",
                    help="청구항 언어 (기본: en). 대화 중에는 /lang 으로 바꿉니다.")
    args = ap.parse_args()

    # 청구항 모드가 기본. --system 이 있으면 그것을, --plain 이면 프롬프트 없이.
    if args.system:
        system, claim_mode = args.system, False
    elif args.plain or CLAIM_SYSTEM is None:
        system, claim_mode = None, False
        if not args.plain and CLAIM_SYSTEM is None:
            print(f"{C_WARN}claim_client.py 를 찾지 못해 일반 대화로 시작합니다.{C_OFF}")
    else:
        system, claim_mode = claim_system_prompt(args.lang), True

    temperature = args.temperature
    if temperature is None:
        # 청구항은 재현성이 중요하므로 낮게, 일반 대화는 원래 값 그대로.
        temperature = 0.2 if claim_mode else 0.7

    prov, base_url, key = resolve_provider(args.provider)

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai 패키지가 없습니다.  pip install openai  을 먼저 실행하세요.")

    # 첫 요청은 콜드 스타트를 기다려야 하므로 타임아웃을 넉넉히.
    timeout = 900.0 if prov["billing"] == "time" else 180.0
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)

    chat = Chat(client, system, prov, temperature=temperature,
                max_tokens=args.max_tokens,
                postprocess=claim_sanitise if claim_mode else None)
    if args.load:
        p = Path(args.load)
        if p.exists():
            chat.load(p)
        else:
            print(f"{C_WARN}파일이 없어 새 대화로 시작합니다: {p}{C_OFF}")

    mode = "청구항 작성" if claim_mode else "일반 대화"
    print(f"\n{C_BOT}Gemma 4 31B{C_OFF} {C_DIM}· {prov['label']} · {mode} · /help 로 명령어 보기{C_OFF}")
    if claim_mode:
        print(f"{C_DIM}도면 여러 장은 figure 순서대로: /image 도면1.png 도면2.png{C_OFF}")
        print(f"{C_DIM}터미널에 끌어다 놓거나, 캡처해서 /clip 으로도 첨부됩니다.{C_OFF}")
    if chat.turns:
        print(f"{C_DIM}이어가기: {chat.turns}턴{C_OFF}")
    print()

    while True:
        try:
            raw = input(f"{C_USER}나{C_OFF}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # 터미널에 도면을 끌어다 놓으면 경로만 적힌 줄이 됩니다. 명령어를 앞에
        # 붙이지 않아도 첨부로 받습니다. POSIX 경로가 /로 시작하므로 아래
        # 명령어 분기보다 먼저 봐야 합니다.
        dropped = dnd_paths(raw)
        if dropped:
            urls = [u for u in (load_image(p) for p in dropped) if u]
            if len(urls) != len(dropped):
                print(f"{C_WARN}  일부 도면을 읽지 못해 중단합니다.{C_OFF}\n")
                continue
            ask_with_images(chat, urls, "")
            continue

        if raw.startswith("/"):
            if raw == "/paste":
                pasted = read_paste()
                if pasted:
                    chat.ask(pasted)
                    print()
                continue
            head, _, rest = raw.partition(" ")
            if head in ("/clip", "/v"):
                ask_with_images(chat, clipboard_images(), rest.strip())
                continue
            if raw.startswith("/image"):
                arg = raw[len("/image"):].strip()
                if not arg:
                    # 경로가 없으면 클립보드를 봅니다. 터미널은 이미지 붙여넣기를
                    # 글자로 받지 못하므로 클립보드를 직접 읽는 수밖에 없습니다.
                    ask_with_images(chat, clipboard_images(), "")
                    continue
                img_paths, question = split_image_args(arg)
                if not img_paths:
                    print(f"{C_WARN}  파일을 찾지 못했습니다: {arg}{C_OFF}\n")
                    continue
                urls = [u for u in (load_image(p) for p in img_paths) if u]
                if len(urls) != len(img_paths):
                    print(f"{C_WARN}  일부 도면을 읽지 못해 중단합니다.{C_OFF}\n")
                    continue
                ask_with_images(chat, urls, question)
                continue
            try:
                if handle_command(chat, raw):
                    print()
                    continue
            except SystemExit:
                break
            print(f"{C_WARN}  모르는 명령어입니다. /help 를 보세요.{C_OFF}\n")
            continue

        chat.ask(raw)
        print()

    if chat.turns:
        print(f"\n{chat.cost_line()}")
        if prov["billing"] == "time":
            print(f"{C_DIM}워커는 {RUNPOD_IDLE_TIMEOUT // 60}분 뒤 자동으로 잠들고 과금이 멈춥니다.{C_OFF}")
    print(f"{C_DIM}종료합니다.{C_OFF}")


if __name__ == "__main__":
    main()
