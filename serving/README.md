# RunPod claim-v3 사용

이 폴더의 기본 실행 경로는 `launch_test.py`(화면)와 `claim_client.py`(CLI)입니다.
둘 다 기존 엔드포인트 `fdiltabt78bogm`의 `claim-v3`를 사용합니다.
DeepInfra나 Hugging Face Jobs를 사용하지 않습니다.

## 설치와 로컬 설정 확인

Windows / Python 3.12:

```powershell
.\setup-test.cmd
# .env의 RUNPOD_API_KEY에 본인 키 입력
.\.venv\Scripts\python.exe serving/launch_test.py --check
```

macOS / Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r serving/requirements-test.txt
cp .env.example .env  # 처음 설정할 때만
# .env에 RUNPOD_API_KEY 입력
.venv/bin/python serving/launch_test.py --check
```

설치 스크립트는 기존 `.env`를 덮어쓰지 않습니다. 프로젝트 `.env`에 입력한 값이
Windows 등에서 상속한 환경변수보다 우선합니다. `.env`가 없거나 값이 비어 있으면
환경변수를 사용합니다. `RUNPOD_ENDPOINT_ID`로 엔드포인트를 변경할 수 있습니다.
`--check`는 키의 존재 여부만 표시하고 값은 출력하지 않으며, 외부 API에 접속하지 않습니다.
로컬 PC에서 필요한 인증은 RunPod API 키뿐입니다. 비공개 HF 모델을 읽는 토큰은
RunPod 워커 템플릿에 보관되어야 하며 Git에 넣지 않습니다.

## 실제로 사용할 때

```powershell
.\start-test.cmd
```

또는 `.venv`의 Python으로 `serving/launch_test.py`를 실행합니다.
7860부터 빈 포트를 찾아 브라우저를 자동으로 엽니다. 기존 화면이 7860을 사용하면
7861 등 다음 빈 포트를 사용합니다. 터미널에 표시된 실제 주소를 확인하세요.
특정 포트를 고정하려면 `--port 7861`로 지정합니다. 고정한 포트는 비어 있어야 합니다.
화면은 localhost에만 바인딩하고 공개 공유 링크를 생성하지 않습니다.
화면을 여는 것만으로 모델을 호출하지 않습니다.

1. 도면을 순서대로 최대 8장, 장당 최대 8MB 업로드합니다.
2. 한국어/영어, 요청 문구와 답변 길이를 선택합니다.
3. **청구항 생성 · 요청 과금 발생**을 누릅니다.
4. 원문 JSON과 실제 응답 모델 ID, 종료 이유를 확인합니다. `length`는 잘린 응답입니다.

CLI도 실행하면 유료 요청을 보냅니다:

```powershell
.\.venv\Scripts\python.exe serving/claim_client.py fig1.png fig2.png --lang ko --max-tokens 1024
```

v3 원문 JSON을 stdout에, 실행 정보를 stderr에 출력합니다. UI와 CLI는 실제 응답의
모델 ID가 요청한 별칭과 다르면 실패 처리합니다. 이는 가중치 자체의 원격 증명은 아닙니다.
기본 응답 대기는 시간 제한이 없습니다. 일시적인 상태 조회 통신 오류는 같은 작업 ID로
재시도하며 추론 요청을 중복 제출하지 않습니다. RunPod 자체의 실행 시간 및 요청 수명은
최대 7일이므로 새 요청에는 두 값을 604800000ms로 지정합니다. 요청 수명에는 배정 대기도
포함됩니다. 완료 또는 서버의 실패·취소·만료 시 대기를 끝냅니다.
Python에서 명시적으로 `iter_job(..., timeout=초)`를 지정하거나 스트림을 종료하면
자신이 제출한 작업의 취소를 시도합니다.
취소 통신이 실패할 수 있고, UI를 닫는 것이 과금 종료 확인을 대신하지는 않습니다.

## 선택 사항: 확정 판례 예제 패널

도면 화면과 CLI는 학습 자료 없이 동작합니다. 판례 어노테이션/자문 예제 패널에는
본인이 보관한 다음 비공개 자료가 추가로 필요합니다. 공개 Git에는 포함하지 않습니다.

- `data/case_rl/releases/rl-pilot-27fc35d39288134b/` 전체 릴리스(manifest 포함)
- `.superloopy/evidence/frozen-release.json` 검증 영수증

두 자료를 같은 상대 경로로 복원하고 화면을 다시 실행하면 확정 예제 패널이 표시됩니다.
이 프로젝트의 기존 작업 폴더에는 이미 설치되어 있습니다. 다른 PC에서는 별도로
보관한 원본을 사용해야 합니다. 클라이언트는 manifest 및 사용 파일의 SHA256을 검사하며,
학습 split의 입력만 노출합니다. 평가 정답은 화면으로 보내지 않습니다.
손상된 자료는 검증을 우회하지 말고 원본에서 다시 복원하세요.

## 운영 상태와 비용

2026-09-13 변경: FlashBoot ON, 엔드포인트 실행 제한 7일, 클라이언트 기본 대기 제한 없음.
기존에 실행한 화면은 재시작해야 새 클라이언트가 적용됩니다. 이미 제출된 작업의 정책은
변경되지 않습니다. 아래 9월 9일 기록은 변경 전 상태입니다.

2026-09-09 마지막 운영 점검: min workers 0, max workers 1, idle timeout 5초,
FlashBoot OFF, 실제 GPU/학습 Pod/네트워크 볼륨 0, 자동 충전 OFF였습니다.
이는 당시 관측이며 실행 시점의 실시간 상태가 아닙니다. 요청을 보내면 워커 시작·처리·
종료 전 대기에 비용이 발생할 수 있습니다. GPU 미배정 또는 긴 첫 로딩으로 실패할 수 있습니다.

`.env`에서 `CLAIM_ENDPOINT_PAUSED=1`로 설정하고 프로그램을 다시 실행하면 이 클라이언트의
요청을 차단합니다. 이 옵션은 RunPod 워커나 다른 클라이언트, 이미 진행 중인 작업을
종료하지 않습니다. 장기 사용 중지는 RunPod 대시보드에서 진행/대기 작업과 실제 워커를
확인하고 max workers를 0으로 변경하세요. 다시 쓸 때는 max 1, min 0으로 설정합니다.

`pinned_lora_bootstrap.py`는 이미 배포된 워커의 revision/hash 검증 진입점입니다.
로컬 UI 실행이나 설치 과정에서 실행되지 않습니다. 모델/템플릿 정보와 검증 한계는
[HANDOFF](../HANDOFF.md)에 있습니다. `claim_chat.py`, `claim_batch.py`, `space/app.py`는
과거 v2/기본 모델용 도구이며 현재 v3 사용 절차는 위 화면과 CLI를 따릅니다.

## 외부 호출 없는 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_serving_v2.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_serverless_setup.py
```

요청을 모킹하여 라우팅, 취소, 잘못된 모델 거부, 설정 우선순위와 설치 독립성을 검사합니다.
실제 서버 가용성이나 생성 품질을 통과했다는 의미는 아닙니다.
