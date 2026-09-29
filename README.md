# da-eval

Contains logic used to evaluate domain adaptation across detector repositories such as
RT-DETRv4, D-FINE and Ultralytics. One implementation of the metrics and the evaluation
protocol is installed into each model repository's environment, so results are directly
comparable across models.

## Protocol

For every source dataset, the checkpoint trained on it is evaluated as follows:

1. **Source `val` split** → precision/recall/F1/F2 vs. confidence curves → the best-F2
   confidence becomes the operating threshold.
2. **Every dataset's `test` split** (source included) → mAP sweep (IoU 0.20–0.95),
   precision/recall/F1/F2 at the source threshold, confidence statistics and image embeddings.
3. **Source `train` split** (optional) → embeddings for the t-SNE.
4. **Domain gap**: unbiased RBF MMD² between source-test and each target-test embedding
   set (median-heuristic bandwidth from the source), plus a combined t-SNE.

Outputs: per-target JSON/NPY files, a CSV per source, and optional W&B logging
(scalars, curves, Table 1 metrics, Table 2 operating points + domain gap, t-SNE).

## Install (inside a model repository)

While developing, with `da-eval` cloned next to the model repositories:

```bash
uv add --editable ../da-eval
```

For final experiments, pin a released version so `uv.lock` records the exact code:

```bash
uv add "da-eval @ git+https://github.com/JosmarSuarezUA/da-eval@v0.1.0"
```

`torch` is not a dependency: the model repository provides it.

## Usage

Datasets are defined once in [`configs/datasets.yaml`](configs/datasets.yaml).
Run from inside the model repository:

```bash
uv run da-eval --model rtdetrv4 \
    --config configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml \
    --datasets ../da-eval/configs/datasets.yaml \
    --checkpoint A=outputs/sds_jp_transfer_rtv4_hgnetv2_s_coco/best_stg1.pth \
    --checkpoint B=outputs/synbase_rtv4_hgnetv2_s_coco/best_stg1.pth \
    --wandb-project rtdetrv4
```

Use `--model dfine` in the D-FINE repository. `uv run da-eval --help` lists all options.
W&B logging needs `WANDB_API_KEY` in the environment or in `~/.env`.

From Python:

```python
from da_eval import load_dataset_configs, run_all_sources
from da_eval.adapters.yaml_engine import RTDETRv4Adapter

adapter = RTDETRv4Adapter(config_path="configs/rtv4/rtv4_hgnetv2_s_coco_custom.yml")
run_all_sources(
    adapter,
    checkpoints={"A": "outputs/.../best_stg1.pth"},
    dataset_configs=load_dataset_configs("../da-eval/configs/datasets.yaml"),
    result_root="results/rtdetr",
)
```

## Adding a model

Subclass `da_eval.pipeline.DetectorAdapter` and implement:

- `_load(checkpoint_path)`: load weights for one source checkpoint.
- `run_split(split, *, min_score, with_predictions, with_embeddings)`: one inference
  pass over `split = {"img_folder", "ann_file"}` returning a `SplitOutput` with
  COCO-format predictions (`image_id`/`category_id` matching the annotation file)
  and optional `(num_images, dim)` embeddings.

Then register it in `da_eval/adapters/__init__.py`.

## Layout

```
da_eval/
  metrics.py            framework-agnostic metrics, plots, CSV and W&B helpers
  pipeline.py           DetectorAdapter interface + evaluation protocol
  adapters/
    yaml_engine.py      RT-DETRv4 and D-FINE (shared YAMLConfig engine)
  cli.py                da-eval command
configs/
  datasets.yaml         shared dataset definitions
```
