#!/usr/bin/env bash
# Space에 배포합니다. 공용 모듈을 매번 다시 복사하므로 원본과 어긋나지 않습니다.
#
#   ./space/deploy.sh Mepeng22/claim-drafter
#
# Space는 미리 만들어 두세요 (huggingface.co/new-space, SDK: Gradio, **Private**).
# 푸시할 때 사용자명과 함께 HF 액세스 토큰(write 권한)을 물어봅니다.
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "사용법: $0 <owner>/<space-name>   예: $0 Mepeng22/claim-drafter" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "$ROOT/serving/claim_prompt.py" "$ROOT/space/claim_prompt.py"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone "https://huggingface.co/spaces/$TARGET" "$WORK/space"
cp "$ROOT"/space/{app.py,claim_prompt.py,requirements.txt,README.md} "$WORK/space/"

cd "$WORK/space"
git add -A
if git diff --cached --quiet; then
  echo "바뀐 것이 없습니다."
  exit 0
fi
git commit -m "Update claim drafter"
git push
echo
echo "배포 완료: https://huggingface.co/spaces/$TARGET"
echo "Space 설정 → Variables and secrets 에서 RUNPOD_API_KEY 를 Secret 으로 넣으세요."
