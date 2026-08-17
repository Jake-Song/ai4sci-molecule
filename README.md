# AI4Sci Molecule

ESOL을 시작점으로 분자 표현과 용해도 회귀를 익히는 프로젝트입니다. 1주차에는
RDKit descriptor baseline을 만들고, 2주차에는 atom embedding과 message passing으로
graph representation을 학습하는 GCN/GIN을 같은 random split에서 비교합니다.
3주차에는 같은 모델을 random split과 Murcko scaffold split에서 seed 5개로 다시
평가해, GNN이 물리적 관계를 배웠는지 아니면 익숙한 molecular motif에 기대고 있는지를
확인합니다.

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

노트북은 다음 세 개입니다.

- [`notebooks/01_molecule_exploration.ipynb`](notebooks/01_molecule_exploration.ipynb):
  descriptor 탐색과 Linear/MLP baseline
- [`notebooks/02_gnn_solubility.ipynb`](notebooks/02_gnn_solubility.ipynb):
  GCN/GIN 학습과 descriptor baseline 비교
- [`notebooks/03_scaffold_split.ipynb`](notebooks/03_scaffold_split.ipynb):
  random split과 scaffold split의 generalization gap 비교

노트북을 처음부터 다시 실행하려면 다음 명령을 사용합니다.

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/02_gnn_solubility.ipynb notebooks/03_scaffold_split.ipynb
```

3주차 노트북은 `results/week3/`에 저장된 sweep 결과를 재사용하므로 보통 몇 초 만에
끝납니다. 캐시를 지우고 다시 계산하면 GPU에서 약 10분이 걸리므로 타임아웃을 넉넉히
잡아 둡니다.

대화형으로 살펴보려면 `uv run jupyter lab`을 실행합니다.

## 결과물

3주차 sweep의 표·원시 예측값·그림은 `results/week3/`에 커밋되어 있습니다
(`metrics.csv`, `predictions.csv`, `summary.json`, `figures/*.png`). 전체를 다시
생성하려면 다음을 실행합니다.

```bash
uv run python -m ai4sci_molecule.week3
```

해석과 한계는 [`reports/week3.md`](reports/week3.md)에 정리해 두었습니다.
`data/`는 다시 내려받을 수 있는 원본이므로 계속 `.gitignore` 대상입니다.

## 테스트

```bash
uv run pytest
```

`tests/test_week1.py`, `tests/test_week2.py`, `tests/test_week3.py`는 네트워크 없이
실행됩니다. 통합 테스트는 이미 내려받은 데이터를 재사용하며, 데이터가 없으면 최초 한 번
ESOL을 다운로드합니다.

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

1·2주차는 고정 seed 42의 동일 index split(ESOL 기준 train 902 / validation 113 /
test 113)을 사용하고, validation RMSE로 early stopping checkpoint를 선택한 뒤 test를
한 번 평가합니다. 3주차는 같은 크기의 scaffold split을 추가로 만들어 두 regime을 seed
5개로 비교합니다. 결과 해석은 [`reports/week3.md`](reports/week3.md)를 참고하세요.
