---
title: 청구항 작성
emoji: 📐
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: test_ui.py
pinned: false
---

# claim-v3 테스트 화면

이 폴더는 프로젝트의 RunPod 서버리스용 Gradio 화면입니다.
로컬 실행은 저장소 루트의 `setup-test.cmd` → `.env` 설정 → `start-test.cmd`를 사용하세요.
[설치·실행 설명](../serving/README.md)을 따릅니다. 화면을 열 때 모델을 호출하지 않으며,
생성 버튼을 누를 때 RunPod 유료 요청을 보냅니다.

기본 모델은 기존 v2에서 추가 강화학습한 `claim-v3`입니다. 도면 근거와 종속항,
판례 어노테이션에서 알려진 품질 문제가 남아 있습니다. 원문과 실제 모델 ID를 표시하고,
다른 모델 응답은 거부합니다. 새로운 서버리스 배포의 실제 응답 검증은 아직 미완료입니다.

`claim_client.py`, `client_config.py`, `claim_prompt.py`는 `serving/`의 복사본입니다.
로컬 UI 의존성은 `serving/requirements-test.txt`로 설치합니다.
`deploy.sh`는 별도 HF Space를 직접 관리하는 경우의 기존 선택 도구로,
현재 RunPod 사용 절차에는 필요하지 않고 자동으로 실행되지 않습니다.
비공개 판례 예제 자료는 공개 Git이나 Space 배포에 포함하지 않습니다.
