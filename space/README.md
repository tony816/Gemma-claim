---
title: 청구항 작성
emoji: 📐
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# 청구항 작성 · Gemma 4 31B

도면을 올리면 독립항 초안을 써 주는 웹 UI입니다. RunPod Serverless에 올려둔
Gemma 4 31B(INT4)를 호출하고, 이 Space는 그 앞단 역할만 합니다.

## 이 Space는 반드시 비공개여야 합니다

공개로 두면 URL을 아는 사람이 소유자의 RunPod GPU를 돌리게 됩니다. 엔드포인트는
시간당 과금이므로 그대로 요금이 됩니다. 생성할 때 **Private** 을 고르세요.

## 설정

Space 설정 → **Variables and secrets**:

| 이름 | 종류 | 값 |
|---|---|---|
| `RUNPOD_API_KEY` | **Secret** | RunPod 콘솔 → Settings → API Keys |
| `RUNPOD_ENDPOINT_ID` | Variable (선택) | 생략하면 `fdiltabt78bogm` |

키는 Secret으로 넣어야 합니다. Variable로 넣으면 Space 설정 화면에 값이 그대로
보입니다. 어느 쪽이든 브라우저로는 내려가지 않지만, Secret만 저장 후 가려집니다.

## 쓰는 법

1. 도면을 올립니다. 여러 장이면 **figure 순서대로** 올리세요 — 그 순서 그대로
   모델에 전달되고, 뒤 도면은 앞 도면을 기준으로 읽힙니다.
2. 질문을 비워두면 "이 도면들의 독립항을 작성해줘"로 나갑니다.
3. 오른쪽 **정리된 청구항** 칸에는 마크다운·`Claim 1:` 라벨·도면 부호·드래프팅
   노트를 걷어낸 결과가 들어갑니다. 모델이 형식을 지켰다면 원문과 같습니다.

**청구항 모드**를 끄면 시스템 프롬프트 없이 일반 대화가 됩니다.

## 비용

첫 질문은 모델 적재로 약 5분, 이후는 5~7초입니다. 마지막 질문 후 5분이 지나면
워커가 잠들고 과금이 멈춥니다. 화면 우측에 누적 비용이 표시됩니다.

같은 세션 안에서 연달아 묻는 편이 훨씬 쌉니다. 5분 넘게 쉬면 다음 질문에서
다시 5분을 기다리게 됩니다.

이 Space 자체(CPU basic)는 무료이며, 48시간 쓰지 않으면 잠들었다가 접속 시
수십 초 안에 깨어납니다.

## 왜 파인튜닝 모델이 아닌가

이 프로젝트에서 학습한 LoRA 어댑터는 출력 형식은 완전히 배웠지만 도면을 읽는
능력을 잃었습니다. 테스트 12건 중 참조와 같은 장치에 대한 청구항을 낸 것은
0건이었습니다. 베이스 모델은 도면을 정확히 읽으므로, 부족했던 형식만 시스템
프롬프트로 채우는 쪽을 택했습니다. 근거는 저장소의
`run_artifacts/POST_HOC_GENERATION_AUDIT.md` 에 있습니다.

## Gradio 버전

`sdk_version` 을 6.26.0 으로 고정했습니다. 이 버전에서 실제로 빌드·동작을
확인했습니다. 6.x 는 `Chatbot(type=)` 과 `Textbox(show_copy_button=)` 을 없앴고
5.x 는 전자가 있어야 dict 형식 메시지를 받으므로, `app.py` 는 설치된 시그니처를
보고 맞춥니다. 버전을 올리더라도 그대로 동작합니다.

## 배포

이 폴더는 [`tony816/Gemma-claim`](https://github.com/tony816/Gemma-claim) 의
`space/` 에서 옵니다. `claim_prompt.py` 는 `serving/claim_prompt.py` 의 복사본이며
`space/deploy.sh` 가 배포할 때마다 다시 복사합니다.
