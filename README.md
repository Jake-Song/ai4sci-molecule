# AI4Sci Molecule

ESOL을 시작점으로 분자 표현과 용해도 회귀를 익히는 프로젝트입니다. 1주차에는
RDKit descriptor와 PyTorch Geometric graph 표현을 비교하고, 같은 random split에서
세 가지 고전 회귀 baseline을 평가합니다.

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

실행 결과가 저장된 노트북은
[`notebooks/01_molecule_exploration.ipynb`](notebooks/01_molecule_exploration.ipynb)입니다.
처음부터 다시 실행하려면 다음 명령을 사용합니다.

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/01_molecule_exploration.ipynb
```

대화형으로 살펴보려면 `uv run jupyter lab`을 실행합니다.

## 테스트

```bash
uv run pytest
```

`tests/test_week1.py`는 네트워크 없이 실행됩니다. 통합 테스트는 이미 내려받은
데이터를 재사용하며, 데이터가 없으면 최초 한 번 ESOL을 다운로드합니다.

재사용 가능한 1주차 API는 `src/ai4sci_molecule/week1.py`에 있습니다.

- `load_esol(root)`
- `smiles_to_graph(smiles, target=None)`
- `calculate_descriptors(smiles_list)`
- `make_random_split(n_samples, seed=42)`
- `build_baselines(seed=42)`
- `regression_metrics(y_true, y_pred)`

고정 seed 42의 index split(ESOL 기준 train 902 / validation 113 / test 113)은
다음 주 GNN 비교에서도 그대로 사용합니다. 이번 범위에는 GNN 학습과 scaffold
split이 포함되지 않습니다.
