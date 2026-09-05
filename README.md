# NeuroMerge

Research artifact for **NeuroMerge: ML-Guided State Merging for Efficient
Symbolic Execution**, accepted to ISSRE 2026.

Authors: Shenghan Zheng, Shitong Zhu, Yu Hao, Xingyu Li, Keyu Man, Zheng Zhang,
Qing Deng, Zhiyun Qian, and Srikanth V. Krishnamurthy (University of California,
Riverside). Shenghan Zheng and Shitong Zhu contributed equally to this work.

NeuroMerge uses machine-learning guidance to select state merges during
symbolic execution. This repository contains the available implementation,
training/evaluation scripts, and a packaged GNN model.

## Repository layout

| Path | Contents |
| --- | --- |
| `klee/` | Modified KLEE symbolic execution engine and its tests/build infrastructure. |
| `MergeGraph/` | ProGraML-derived program graph construction code. |
| `scripts/utils.py` | Graph generation, dataset processing, and experiment utilities. |
| `scripts/ggnn/` | GNN data processing, modeling, and training code. |
| `scripts/rf_model_train_eval.py` | Random Forest training and evaluation script. |
| `scripts/eval.py`, `scripts/eval_qce.py` | Evaluation drivers. |
| `gnn_model/mergegraph_gnn.mar` | Packaged GNN checkpoint for the customized TorchServe handler. |
| `serve/serve-master/` | Vendored TorchServe with the MergeGraph GNN handler. |
| `neuse_env.yml`, `requirements.txt` | Recorded Python/Conda environment specifications. |

The source layout and implementation are preserved; this release starts with
a new initial commit and does not import the previous repository's Git history.
`NeuSE` and `neuse` remain in some internal paths and environment names.

## Environment and setup

The recorded environment uses Ubuntu 20.04, Python 3.9, and CUDA 11.3-era
dependencies. It is an environment snapshot, not a tested installation recipe
for current operating systems. Use an isolated development environment.

```bash
git clone https://github.com/seclab-ucr/NeuroMerge.git
cd NeuroMerge
conda env create --file neuse_env.yml
conda activate neuse
```

The customized model server is under `serve/serve-master/`, not directly under
`serve/`. Its installation also requires the Java/Gradle prerequisites described
in its [upstream build documentation](serve/serve-master/README.md).

```bash
python -m pip install ./serve/serve-master/
```

KLEE and the modified graph construction component need separate native builds.
See their checked-in build files and the component documentation in
[`klee/`](klee/) and [`MergeGraph/`](MergeGraph/). Installing an unmodified
ProGraML or TorchServe package is not a substitute for the customized source.

## Models, data, and experiment prerequisites

The included GNN archive contains a serialized checkpoint and a model wrapper.
Its manifest identifies the model as `mergegraph0429`; the archive filename is
`mergegraph_gnn.mar`. The customized handler is
[`ggnn_classifier.py`](serve/serve-master/ts/torch_handler/ggnn_classifier.py).

The current snapshot does **not** include the experiment `dataset/` directory,
the trained RF `.joblib` model, benchmark bitcode, train/validation/test splits,
or the handler's `dataset/vocab/programl.csv`. Vendored component test fixtures
are not the paper's evaluation dataset. These experiment inputs must be supplied
before training, inference, or end-to-end reproduction; the GNN archive alone
is not a standalone demo.

Before running experiments, configure the original machine-specific paths:

- `scripts/utils.py`: KLEE build locations, graph tool location, and dataset paths.
- `scripts/eval.py`, `scripts/eval_qce.py`, and `scripts/rf_model_train_eval.py`:
  local experiment and output paths.
- `scripts/ggnn/run.py`: dataset selection and paths.
- `serve/serve-master/ts/torch_handler/ggnn_classifier.py`: `REPO_ROOT` and
  vocabulary path (the current default is `/data/<username>/NeuSE`).
- `scripts/klee-test.env`: environment passed to the analyzed programs.

After supplying the missing inputs, configuring paths, and building the required
components, the corresponding model-serving command from the repository root is:

```bash
torchserve --start \
  --model-store gnn_model \
  --models mergegraph0429=mergegraph_gnn.mar \
  --ts-config serve/serve-master/ts.config \
  --foreground
```

The checked-in configuration uses inference port `50051`; the matching prediction
endpoint is `/predictions/mergegraph0429`. It binds to all interfaces, so restrict
it to a trusted, isolated environment or change the bind address before use.
Only load trusted serialized models. This migration has not rerun model training
or the paper's end-to-end experiments.

Review experiment scripts before execution: some launch multiple processes,
terminate KLEE processes, or remove generated data. Do not run cleanup tasks on
directories containing data you want to keep.

## Citation

Please cite the ISSRE 2026 paper, *NeuroMerge: ML-Guided State Merging for
Efficient Symbolic Execution*. Proceedings metadata and a DOI can be added once
available.

## Licenses and contact

Third-party license files and copyright notices are retained in place; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This repository does not apply
a new blanket license to the mixed-source tree.

For artifact questions or missing experiment inputs, please open an issue in
this repository or contact Shitong Zhu at `szhu014@ucr.edu`.
