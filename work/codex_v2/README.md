# codex_v2 validator

기존 파일을 건드리지 않고 현재 최종 recipe를 재현하는 별도 검증 도구입니다.

핵심 차이:

- Trackman prior를 실제 feature build 전에 부착합니다.
- 최종 `TRAIN_FROM=2023`을 2시즌 rolling window로 이동해 2023/2024를 검증합니다.
- LGBM 5-seed, Logistic, MLP1, quantile-normal MLP2를 최종 logit 가중치 그대로 섞습니다.
- count 보정과 `DRIFT-D_AUTO`, R/F별 logit 평균 shift를 모두 포함합니다.
- 기본 실행은 학습 없는 dry-run입니다. `--fit` 없이는 큰 학습이 시작되지 않습니다.

## 빠른 정합성 검사

```powershell
python work/codex_v2/validate_recipe_v2.py `
  --model-meta work/model/meta.json `
  --train-script "C:\Users\noc07\OneDrive\Desktop\류민기\work\train_final.py"
```

현재 `codex` clone의 `train_final.py`는 MLP3가 켜진 과거 snapshot이므로, 최신 원본을
감사하려면 위처럼 `--train-script` 경로를 명시합니다. 스크립트는 실행하지 않고 AST로
literal 설정만 읽습니다.

## 데이터·피처·fold dry-run

```powershell
python work/codex_v2/validate_recipe_v2.py `
  --data data/train.csv `
  --trackman-prior work/tm_prior.parquet `
  --pitcher-map work/idmap_pitcher_v2.parquet `
  --model-meta work/model/meta.json
```

## 실제 rolling 검증

```powershell
python work/codex_v2/validate_recipe_v2.py `
  --fit `
  --data work/train.parquet `
  --trackman-prior work/tm_prior.parquet `
  --pitcher-map work/idmap_pitcher_v2.parquet `
  --eval-years 2023,2024 `
  --output work/codex_v2/results/current_recipe.json
```

`--quick`은 모델별 seed를 1개만 써서 경로를 확인하는 옵션이며 정식 점수로 사용하면 안 됩니다.

## 단위 테스트

```powershell
python -m unittest discover -s work/codex_v2 -p "test_*.py" -v
python -m unittest discover -s work/codex_v2/tests -p "test_*.py" -v
```

번들·모델 산출물의 allowlist/해시/입출력 검증은 `manifest_verify.py`의
`check-model`, `build`, `verify-bundle`, `verify-input`, `verify-output` 명령을 사용합니다.

## 격리된 후보 실험

- `trackman_prior_v2.py`: 피처별 유효 분모를 사용하는 Trackman 후보 생성기
- `previous_season_pitcher.py`: 직전 시즌 R 투수 EB prior 생성기
- `evaluate_prev_pitcher_residual.py`: 현 2-MLP OOF 잔차에 대한 prior 교차 이전 검사
- `team_residual.py`: OOF에서만 학습하는 수축 group residual 보정
- `evaluate_team_residual.py`: batter-team 보정의 연도 교차 이전 및 team-cluster 검사
- `evaluate_team_residual_v2.py`: 고정 가중치의 연도·반시즌·forward 이전 재검증
- `TEAM_RESIDUAL_V2_RESULT.md`: 최종 채택 근거, 배포 설정, 제출 ZIP 검증 기록
- `team_submission/`: 팀 잔차 artifact 생성, 격리 후보 build, runtime 보정
- `EXPERIMENT_PLAN.md`: 채택 기준과 지금까지의 기각/보류 결과

후보 도구는 어느 것도 champion 모델 파일을 자동으로 변경하지 않습니다.
