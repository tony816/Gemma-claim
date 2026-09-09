# Gemma 31B — 청구항·판례 서버리스 클라이언트

기존 v2에서 추가 강화학습한 **`claim-v3`**를 RunPod Serverless로 호출하는 로컬
테스트 화면과 CLI입니다. 기본 엔드포인트는 `fdiltabt78bogm`입니다.
추론은 RunPod에서 실행하며, PC에는 모델 가중치나 CUDA를 설치하지 않습니다.

## 나중에 사용하기

Python 3.12와 Git이 필요합니다. Windows에서는:

```powershell
git clone https://github.com/tony816/Gemma-claim.git
cd Gemma-claim
.\setup-test.cmd
# 생성된 .env에 본인의 RUNPOD_API_KEY를 입력
.\.venv\Scripts\python.exe serving/launch_test.py --check
# 실제로 사용할 때만 화면 실행
.\start-test.cmd
```

화면은 7860부터 빈 포트를 찾아 브라우저에 자동으로 열립니다. 해당 화면에서
도면을 올리고 **청구항 생성**을 누르면 유료 요청을
보냅니다. 설치, `--check`, 화면을 여는 동작에는 모델 호출이 없습니다.
`--check`는 로컬 설정만 확인하며 서버의 가용성을 확인하지 않습니다.
키와 비공개 모델에 대한 권한은 Git clone으로 제공되지 않습니다.

[실행 방법·선택적 판례 자료 설치](serving/README.md) · [현재 상태와 한계](HANDOFF.md)

## 모델과 검증 범위

- 비공개 어댑터: `Mepeng22/gemma-4-31b-claim-rl-v3`
- 고정 가중치 revision: `e5aac01e9fe0af54b17e7f52b66486112183035e`
- 기존 v2는 별도 비공개 저장소에 보존됩니다.
- HF 재다운로드 및 전체 GPU 로딩은 이전 실행에서 확인했습니다. 최근 서버리스
  재배포는 설정 변경까지 확인했고, 워커 미배정으로 새 응답은 확인하지 못했습니다.
- 파일럿에서 도면 근거·종속항·판례 어노테이션 오류가 남았습니다. 자동 점수를
  법률적 정확도나 전반적인 품질 향상으로 해석하면 안 됩니다.

아래는 원래 SFT 학습 파이프라인의 설명입니다. 서버리스 화면 설치에는 필요하지 않습니다.

## Original training setup

Vision-language fine-tuning of **`google/gemma-4-31B-it`** on a frozen, 114-record
multi-image dataset of patent claim pages. Given one or more claim-page images in
order, the model drafts a single independent physical apparatus / device / system /
cartridge / assembly claim.

The dataset (`v1.1.2-independent-oracle-clean`) is private and is not in this
repository. It is treated as an immutable reference release: nothing here rewrites,
re-splits, or filters it.

## Layout

```
pipeline/
  common.py         paths, seeding, schema normalisation, model loading
  fetch_dataset.py  download + SHA-256 gate + extract + integrity manifest
  preflight.py      data-contract gate (counts, images, targets, leakage)
  token_audit.py    per-record token lengths; proves zero target truncation
  dataset.py        encoding + assistant-only loss masking + collator
  train.py          bf16 LoRA SFT with a pre-training base baseline
  evaluate.py       base vs tuned under identical conditions
  push_hub.py       publish adapter and merged model
  inference.py      standalone image(s) -> claim
  report.py         FINAL_TRAINING_REPORT.md + status flags
run_all.sh          pod-side orchestrator (resumable stages)
reproduce.sh        end-to-end reproduction from a dataset ZIP
tests/              offline contract + masking tests (no GPU, no network)
```

## Method

**bf16 LoRA on the language tower, vision tower frozen.**

91 training records against ~31B parameters is well inside the regime where
full-weight updates memorise instead of generalise, and 62.5 GB of bf16 weights
plus Adam moments does not fit any single available GPU. LoRA keeps the trainable
count ~4 orders of magnitude smaller, keeps the run resumable, and leaves the
pretrained visual features untouched.

No quantisation: at 141 GB of VRAM the weights fit in bf16, so QLoRA would add
quantisation error and a dependency on 4-bit kernels being correct for a
recently-released architecture, for no capacity benefit.

### Loss masking

The mask is derived structurally, not by pattern-matching the chat template. For
each record the generation prompt is rendered and asserted to be a strict prefix of
the full rendered conversation, and to tokenise to a strict prefix of the full token
ids. Everything up to that boundary — system text, user text, image placeholders —
is set to `-100`, as is padding. Only assistant tokens carry loss. If the prefix
property ever fails, the run stops rather than silently masking the wrong span.

### Hard failures

The pipeline stops rather than degrading on: NaN/Inf loss, target truncation,
missing images, image-order contract violations, empty assistant targets, records
appearing in more than one split, and any reference to oracle / canonical /
excluded / evaluation material from a training input.

## Data contract

- Model input is limited to `hf_multimodal/{train,validation,test}.jsonl` and the
  `images/` files those records reference.
- `metadata` is never fed to the model.
- `images[]` order is preserved; no image is dropped.
- Splits are used as shipped; validation and test are never folded into train.
- Test is read once, after the configuration and best checkpoint are frozen.

## Run

On a GPU host with the dataset ZIP in hand:

```bash
./reproduce.sh /path/to/final_dataset_v112.zip [output_dir]
```

Offline tests (no GPU, no network, no base model):

```bash
python tests/test_pipeline.py
```

## Environment

| variable | purpose |
|---|---|
| `HF_TOKEN` | Hub token with write access; required only for `push_hub.py` |
| `DATASET_URL` | direct dataset URL; otherwise `gdown` uses `DRIVE_FILE_ID` |
| `BASE_MODEL`, `BASE_MODEL_REVISION` | pin the base model |
| `EPOCHS`, `LR`, `MICRO_BS`, `GRAD_ACCUM`, `LORA_R`, `SEED` | training knobs |
| `HF_ADAPTER_REPO`, `HF_MERGED_REPO`, `HF_PRIVATE` | publication targets |

Tokens are read from the environment only. They are never written to artifacts,
logs, model cards, or this repository.

## Licence

Derivatives of `google/gemma-4-31B-it` remain subject to the Gemma Terms of Use,
including its use restrictions, and must carry those terms downstream.
