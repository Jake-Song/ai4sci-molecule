# AI4Sci Molecule

ESOL을 시작점으로 분자 표현과 용해도 회귀를 익히는 프로젝트입니다. 1주차에는
RDKit descriptor baseline을 만들고, 2주차에는 atom embedding과 message passing으로
graph representation을 학습하는 GCN/GIN을 같은 random split에서 비교합니다.
3주차에는 같은 모델을 random split과 Murcko scaffold split에서 seed 5개로 다시
평가해, GNN이 물리적 관계를 배웠는지 아니면 익숙한 molecular motif에 기대고 있는지를
확인합니다. 4주차에는 새로 학습하지 않고 3주차의 분자 단위 예측을 13개 molecular property와
대조해, prediction error를 데이터에 대한 진단 도구로 씁니다.

## 설치

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.
기본 환경은 PyTorch의 공식 CUDA 13 wheel을 사용합니다. CUDA 13을 지원하는
NVIDIA 드라이버가 설치된 Linux/WSL x86-64 환경에서 다음 명령을 실행합니다.

```bash
uv sync
```

GPU가 없는 환경에서는 `gpu` 그룹을 끄고 CPU-only wheel을 설치합니다.

```bash
uv sync --no-group gpu --group cpu
```

CPU 환경에서 명령을 실행할 때도 같은 그룹 선택을 지정합니다.

```bash
uv run --no-group gpu --group cpu <command>
```

첫 데이터 로드 시 PyG가 ESOL 원본을 내려받아 `data/MoleculeNet/`에 저장합니다.
`data/`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다.

## 노트북 실행

노트북은 다음 네 개입니다.

- [`notebooks/01_molecule_exploration.ipynb`](notebooks/01_molecule_exploration.ipynb):
  descriptor 탐색과 Linear/MLP baseline
- [`notebooks/02_gnn_solubility.ipynb`](notebooks/02_gnn_solubility.ipynb):
  GCN/GIN 학습과 descriptor baseline 비교
- [`notebooks/03_scaffold_split.ipynb`](notebooks/03_scaffold_split.ipynb):
  random split과 scaffold split의 generalization gap 비교
- [`notebooks/04_error_analysis.ipynb`](notebooks/04_error_analysis.ipynb):
  prediction error와 molecular property의 관계, 모델 간 오차 일치도

노트북을 처음부터 다시 실행하려면 다음 명령을 사용합니다.

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/02_gnn_solubility.ipynb notebooks/03_scaffold_split.ipynb \
  notebooks/04_error_analysis.ipynb
```

3주차 노트북은 `results/week3/`에 저장된 sweep 결과를 재사용하므로 보통 몇 초 만에
끝납니다. 캐시를 지우고 다시 계산하면 GPU에서 약 10분이 걸리므로 타임아웃을 넉넉히
잡아 둡니다. 4주차 노트북은 학습을 하지 않고 3주차 예측만 읽으므로 CPU에서 1분 이내에
끝납니다.

대화형으로 살펴보려면 `uv run jupyter lab`을 실행합니다.

## 결과물

3주차 sweep의 표와 그림은 `results/week3/`에 커밋되어 있습니다
(`metrics.csv`, `summary.json`, `splits.json`, `figures/*.png`). 분자 단위 예측값인
`predictions.csv`는 1 MB에 가까워 `.gitignore` 대상이며, 아래 명령으로 나머지 결과와
함께 다시 만들 수 있습니다(GPU 기준 약 10분).

```bash
uv run python -m ai4sci_molecule.week3
```

4주차 분석의 표와 그림은 `results/week4/`에 커밋되어 있습니다. 분자 단위
`molecule_errors.csv`는 1.1 MB로 `.gitignore` 대상이며, 아래 명령으로 나머지 결과와 함께 다시
만듭니다. 학습이 없으므로 3주차 캐시가 있으면 CPU에서 1분 이내에 끝납니다.

```bash
uv run python -m ai4sci_molecule.week4
```

노트북과 `load_or_run_sweep` / `load_or_run_error_analysis`는 캐시 파일이 하나라도 없거나
설정이 다르면 자동으로 다시 계산하므로, 새로 clone한 저장소에서도 그대로 실행됩니다.

해석과 한계는 [`reports/week3.md`](reports/week3.md)와
[`reports/week4.md`](reports/week4.md)에 정리해 두었습니다.
`data/`도 다시 내려받을 수 있는 원본이므로 계속 `.gitignore` 대상입니다.

## 테스트

```bash
uv run pytest
```

`tests/test_week1.py`, `tests/test_week2.py`, `tests/test_week3.py`,
`tests/test_week4.py`는 네트워크 없이 실행됩니다. 통합 테스트는 이미 내려받은 데이터를
재사용하며, 데이터가 없으면 최초 한 번 ESOL을 다운로드합니다.

재사용 가능한 1주차 API는 `src/ai4sci_molecule/week1.py`에 있습니다.

- `load_esol(root)`
- `smiles_to_graph(smiles, target=None)`
- `calculate_descriptors(smiles_list)`
- `make_random_split(n_samples, seed=42)`
- `build_baselines(seed=42)`
- `regression_metrics(y_true, y_pred)`

재사용 가능한 2주차 API는 `src/ai4sci_molecule/week2.py`에 있습니다.

- `GCNRegressor(...)`
- `GINRegressor(...)` (bond feature를 사용하는 GINE 구현)
- `TrainingConfig(...)`
- `build_gnn_models(...)`
- `train_gnn(model, dataset, split, ...)`
- `predict_gnn(result, dataset, indices, ...)`

재사용 가능한 3주차 API는 `src/ai4sci_molecule/week3.py`에 있습니다.

- `murcko_scaffold(smiles, ...)` / `scaffold_groups(smiles_list, ...)`
- `make_scaffold_split(smiles_list, seed=None, ...)` — `make_random_split`과 동일한
  `{"train", "validation", "test"}` 계약을 따릅니다
- `scaffold_statistics(...)` / `scaffold_table(...)` / `split_scaffold_overlap(...)`
- `nearest_neighbour_similarity(query_smiles, reference_smiles, ...)`
- `target_shift_table(targets, splits)`
- `SweepConfig(...)` / `run_generalization_sweep(dataset, ...)` / `load_or_run_sweep(...)`
- `aggregate_metrics(...)` / `generalization_gap(...)` /
  `similarity_error_correlation(...)` / `bin_similarity_errors(...)`
- `plot_*(...)` — 모두 `Figure`를 반환하므로 노트북에서 표시하고 `save_figures(...)`로
  같은 그림을 저장합니다

재사용 가능한 4주차 API는 `src/ai4sci_molecule/week4.py`에 있습니다. 학습을 하지 않으므로
`torch`를 import하지 않고, 3주차 예측 프레임만 입력으로 받습니다.

- `molecular_properties(smiles_list)` / `property_table(predictions)` — 1주차 descriptor
  5개를 재사용해 13개로 확장합니다
- `pool_split_types(predictions, ...)` / `effective_sample_sizes(predictions)` —
  `random → in_domain`, `scaffold`·`scaffold_shuffled` → `out_of_scaffold`
- `build_error_profile(predictions, ...)` — seed를 평균한 분자 단위 표.
  `residual = predicted − actual`이고, `Dummy (mean)`의 오차를 내재적 난이도
  (`baseline_absolute_error`)로 함께 실어 줍니다
- `rank_correlation(...)` / `partial_correlation(...)` /
  `bootstrap_correlation_interval(...)` — 퇴화 입력에서는 예외 대신 `nan`을 돌려줍니다
- `property_error_correlation(...)` — ρ + bootstrap CI + Dummy 통제군 + partial ρ +
  `trained_on` 순환성 플래그
- `bin_property_errors(profile, property_name, ...)` / `property_bin_summary(...)` —
  사분위 구간별 ratio-of-means 정규화
- `error_property_importance(...)` — held-out $R^2$를 `trustworthy` 플래그와 함께 먼저
  보고하고, permutation importance는 held-out fold에서 측정합니다
- `model_error_agreement(...)` / `model_error_disagreement(...)` — 같은 표현 family끼리
  같은 분자에서 틀리는지 확인합니다
- `shrinkage_fit(...)` / `scaffold_error_table(...)` / `case_studies(...)`
- `AnalysisConfig(...)` / `run_error_analysis(...)` / `load_or_run_error_analysis(...)` /
  `build_figures(...)` / `summarise(...)`
- `plot_*(...)` — 3주차와 같은 계약으로 `Figure`를 반환합니다

1·2주차는 고정 seed 42의 동일 index split(ESOL 기준 train 902 / validation 113 /
test 113)을 사용하고, validation RMSE로 early stopping checkpoint를 선택한 뒤 test를
한 번 평가합니다. 3주차는 같은 크기의 scaffold split을 추가로 만들어 두 regime을 seed
5개로 비교합니다. 4주차는 그 결과를 재사용해 분자 단위 오차를 property와 대조합니다.
결과 해석은 [`reports/week3.md`](reports/week3.md)와
[`reports/week4.md`](reports/week4.md)를 참고하세요.
