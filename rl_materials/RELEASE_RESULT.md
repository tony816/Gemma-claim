# 강화학습 재료 확정본

기존 파인튜닝된 Gemma v2에 이어 학습할 **강화학습 재료의 1차 확정본**입니다.
한국·미국 판례, 도면을 근거로 한 청구항 세트, 판례 어노테이션, 특허 자문을 다룹니다.
GPU·모델 가중치·모델 학습은 사용하지 않았고, 기존 v2는 그대로입니다.

| 구성 | 학습 | 채점 검증 | 평가 | 합계 |
|---|---:|---:|---:|---:|
| 에피소드 | 88 | 20 | 20 | 128 |
| 청구항 작성 | 32 | 8 | 8 | 48 |
| 판례 기반 자문 | 56 | 12 | 12 | 80 |
| 선호 비교 | 128 | 24 | 24 | 176 |
| 판례 어노테이션 포함 | 8 | 4 | 4 | 16 |

검토한 판례는 한국 20건·미국 20건입니다. 각 관할에서 14건을 학습,
3건을 채점 검증, 3건을 평가로 나눴습니다. 청구항 세트마다 독립항 1개와
종속항 2개가 있어 총 독립항 48개·종속항 96개입니다.

내용은 작성자와 분리된 AI 검토를 거쳐 원문 페이지·발언 주체·도면 근거를
확인했습니다. 원본 시험 자료 75건은 보존했고, 신규 청구항 평가 8건은
기존 특허 계열과 분리했습니다. 전체 판례 DB를 의미적으로 검토했다는 뜻은
아니며, 이번에 선별하고 검토한 범위의 확정본입니다.

블라인드 채점은 24개 비교에서 검토된 선호 판단과 24개 모두 일치했습니다.
이는 기록된 AI 채점 방식과 이 비교 집합의 검증 결과입니다. 강화학습된
젬마의 성능 점수가 아닙니다. CPU에서 전체 128개 입력·참조 답안을
젬마 프로세서로 인코딩했고, 잘림 없이 최대 7,003토큰이었습니다.

## 파일 사용

- [확정본 설명](<C:/Users/VIEW LIFW/Projects/Gemma-claim/data/case_rl/releases/rl-pilot-27fc35d39288134b/README.md>)
- [완료 검사 결과](<C:/Users/VIEW LIFW/Projects/Gemma-claim/data/case_rl/releases/rl-pilot-27fc35d39288134b/MATERIAL_STATUS.json>)
- [학습용 모델 입력](<C:/Users/VIEW LIFW/Projects/Gemma-claim/data/case_rl/releases/rl-pilot-27fc35d39288134b/policy/train.jsonl>)
- [보상용 참조 답안](<C:/Users/VIEW LIFW/Projects/Gemma-claim/data/case_rl/releases/rl-pilot-27fc35d39288134b/reward/references.train.jsonl>)
- [학습용 선호 비교](<C:/Users/VIEW LIFW/Projects/Gemma-claim/data/case_rl/releases/rl-pilot-27fc35d39288134b/reward/preferences.train.jsonl>)
- [한국어 사용법](<C:/Users/VIEW LIFW/Projects/Gemma-claim/rl_materials/HOW_TO_USE_RL_MATERIALS.md>)

모델에는 `policy`의 메시지와 이미지를 전달합니다. `reward`의 답안·선호 표지와
`audit`의 검토 자료는 보상 평가용으로 분리합니다. 어노테이션의 판례 정보는
실제로 제공된 카드의 사건번호·법원·선고일·페이지에 연결됩니다. 모델 가중치
안의 인과관계를 추정하지 않습니다.

확정본에는 236개 이미지 자산을 포함해 283개 파일이 들어 있으며, 전체 파일
해시가 검증됐습니다. 내용 변경은 기존 확정본을 덮어쓰지 않고 새 버전으로
검토·확정해야 합니다. 모델의 실제 능력과 GPU에서의 강화학습 실행은 후속 작업입니다.

최종 종합 검토도 승인됐습니다. [검토 보고서](<C:/Users/VIEW LIFW/Projects/Gemma-claim/.superloopy/evidence/jinbe-final-gate-report.md>)에서 내용 검토·코드 검토·QA·완료 조건의 확인 결과를 볼 수 있습니다.
