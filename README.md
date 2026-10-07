# da-eval

Shared training launcher and domain-adaptation evaluation for object detectors
(RT-DETRv4, D-FINE; Ultralytics planned). It is installed automatically as a dependency of
each model repository, so you normally don't clone it yourself. One implementation of the
metrics and the protocol is used for every model, so results are directly comparable.

## 1. Describe your datasets in one YAML file

Copy [`datasets.example.yaml`](datasets.example.yaml) **outside the repository** and fill in
your paths. Each dataset is either COCO (annotation JSON + image folder per split) or YOLOv5
(a `data.yaml`):

```yaml
sds:                                    # name used in commands: --source sds
  label: SeaDronesSee                   # optional display name
  train: {img_folder: /data/sds/images/train, ann_file: /data/sds/train.json}
  val:   {img_folder: /data/sds/images/val,   ann_file: /data/sds/val.json}
  test:  {img_folder: /data/sds/images/test,  ann_file: /data/sds/test.json}
  iou: 0.20                             # optional (default 0.50)
  min_score: 0.001                      # optional (default 0.001)

afo:
  yolo: /data/afo/data.yaml             # YOLOv5: splits and class names read from data.yaml
```

- Paths are absolute or relative to the YAML file. No particular folder layout is required.
- Splits: **training** needs `train` + `val`; an **evaluation source** needs `val` + `test`;
  a **target** needs `test`. `train` is optional for evaluation (only used for the t-SNE).
- YOLOv5 datasets are converted to COCO once and cached in `~/.cache/da-eval`
  (change with `DA_EVAL_CACHE`). The cache is rebuilt automatically when images or labels change.
- Category ids can be anything; classes are ordered by sorted category id everywhere.

Check the file before a long run (paths, COCO files, missing images, class lists):

```bash
uv run da-eval check --datasets /path/to/my_datasets.yaml
```

## 2. Train

From the model repository, with any of its model configs:

```bash
uv run da-train --config configs/rtv4/rtv4_hgnetv2_s_coco.yml \
    --datasets /path/to/my_datasets.yaml --source sds --gpus 0 \
    -- --use-amp --seed 0
```

- Writes zero-based copies of the source's train/val annotations to
  `<output-dir>/annotations/` and runs `torchrun train.py`, overriding only the dataset paths,
  `num_classes` and the output directory. The model, schedule and augmentations come from the config.
- Output goes to `output/<config name>_<source>/` (change with `--output-dir`), together with
  `da_train.json` recording the dataset files, classes and exact command.
- Everything after `--` is passed to `train.py`, e.g. `-t pretrained.pth` or
  `-u epochs=100`. `--gpus 0,1` trains on two GPUs; `--dry-run` only prints the command.

## 3. Evaluate

Every checkpoint is evaluated on the `test` split of every dataset in the YAML:

```bash
uv run da-eval --model rtdetrv4 --config configs/rtv4/rtv4_hgnetv2_s_coco.yml \
    --datasets /path/to/my_datasets.yaml \
    --checkpoint sds=output/rtv4_hgnetv2_s_coco_sds/best_stg1.pth \
    --checkpoint afo=output/rtv4_hgnetv2_s_coco_afo/best_stg1.pth \
    --wandb-project my_project            # optional
```

Use `--model dfine` in the D-FINE repository. Results go to `results/<model>/source_<label>/`.
W&B logging needs `WANDB_API_KEY` in the environment or in `~/.env`.

The number of classes is taken from the datasets, so a 1-class checkpoint can be evaluated with
an 80-class COCO config. ImageNet backbone weights are not loaded or downloaded during evaluation,
because the checkpoint replaces every weight. Config keys can be overridden like `train.py -u`
(`-u key=value ...`).

### Protocol

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

## From Python

```python
from da_eval import load_dataset_configs, run_all_sources
from da_eval.adapters.yaml_engine import RTDETRv4Adapter

adapter = RTDETRv4Adapter(config_path="configs/rtv4/rtv4_hgnetv2_s_coco.yml")
run_all_sources(
    adapter,
    checkpoints={"sds": "output/rtv4_hgnetv2_s_coco_sds/best_stg1.pth"},
    dataset_configs=load_dataset_configs("/path/to/my_datasets.yaml"),
    result_root="results/rtdetr",
)
```

## Adding a model

Subclass `da_eval.pipeline.DetectorAdapter` and implement:

- `_load(checkpoint_path)`: load weights for one source checkpoint (`self.num_classes`
  holds the source dataset's number of classes).
- `run_split(split, *, min_score, with_predictions, with_embeddings)`: one inference
  pass over `split = {"img_folder", "ann_file"}` returning a `SplitOutput` with
  COCO-format predictions (class index i → i-th category in sorted id order of the
  annotation file) and optional `(num_images, dim)` embeddings.

Then register it in `da_eval/adapters/__init__.py`.

## Developing da-eval

Model repositories install a tagged release from GitHub. To test local changes, install your
clone into the model repository's environment and skip the automatic re-sync:

```bash
uv pip install -e /path/to/da-eval
uv run --no-sync da-eval ...
```

## Layout

```
da_eval/
  datasets.py           datasets YAML (COCO / YOLOv5), checks, zero-based training annotations
  metrics.py            framework-agnostic metrics, plots, CSV and W&B helpers
  pipeline.py           DetectorAdapter interface + evaluation protocol
  adapters/
    yaml_engine.py      RT-DETRv4 and D-FINE (shared YAMLConfig engine)
  cli.py                da-eval command (evaluation, check)
  train.py              da-train command
datasets.example.yaml   template for your datasets file
```
