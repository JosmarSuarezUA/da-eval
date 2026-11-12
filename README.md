# da-eval

Shared training launcher and domain-adaptation evaluation for object detectors
(RT-DETRv4, D-FINE, Ultralytics YOLO). It is installed automatically as a dependency of
each model repository, so you normally don't clone it yourself. One implementation of the
metrics and the protocol is used for every model, so results are directly comparable.
`da-train` covers the RT-DETRv4 / D-FINE repositories; YOLO models are trained with Ultralytics
and only evaluated here.

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

### Ultralytics YOLO (e.g. YOLO26)

The `.pt` checkpoint contains the model, so no config or model repository is needed. In any
project (Python ≥ 3.10) with the yolo extra:

```bash
uv add "da-eval[yolo,analysis] @ git+https://github.com/JosmarSuarezUA/da-eval@<tag>"
uv run da-eval --model yolo --datasets /path/to/my_datasets.yaml \
    --checkpoint sds=runs/detect/sds/weights/best.pt \
    --checkpoint afo=runs/detect/afo/weights/best.pt \
    --wandb-project my_project
```

- **Classes:** YOLO class i must be the i-th category (sorted by id) of the source dataset;
  the checkpoint's class count is checked against it.
- **Input:** Ultralytics' own preprocessing, a **square letterbox** (aspect ratio kept, grey
  padding) at the checkpoint's training `imgsz`; `--imgsz` overrides it. (`predict()` would use a
  rectangular letterbox only for batches of same-sized images; the square one makes results
  independent of `--batch-size`.) RT-DETRv4 / D-FINE stretch to 640×640 instead.
- **Boxes:** YOLO26 is NMS-free like RT-DETR; up to 300 boxes per image (`max_det`), score ≥ 0.001.
- **Embeddings:** by default the inputs of the `Detect` head, the neck outputs P3/P4/P5 (64/128/256
  channels for YOLO26n, 448-d concatenated), the counterpart of the RT-DETR encoder output.
  `--levels P3 P4 P5` works as for the other models. `--embedding-module model.<i>` hooks another layer.
- Results go to `results/yolo/source_<label>/` (`--result-root` to separate e.g. yolo26n and yolo26s).
- **Same outputs as RT-DETRv4 / D-FINE:** the same files, CSV columns, W&B sections, summary keys
  and tables. Only the model identifiers differ: `model_name` (e.g. `YOLO26n`), the tag and the
  artifact name (`embeddings-yolo-<source>`). `config_path` is the architecture file stored in the
  checkpoint (e.g. `yolo26n.yaml`).
- Every model records its preprocessing in the W&B run config and the embeddings manifest:
  `input_size` ([h, w]) and `resize` (`stretch` for RT-DETRv4 / D-FINE, `letterbox` for YOLO).

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
Each embeddings `.npy` has a `<name>_images.csv` mapping rows to `image_id`/`file_name`, and
each source folder an `embeddings_manifest.json`; with W&B they are also logged as the
artifact `embeddings-<model>-<source>`.

## 4. Analyse the embeddings

Projections and pymdma distribution metrics of the embeddings saved by the evaluation,
for one source model (needs the analysis extra, `pip install "da-eval[analysis]"`):

```bash
uv run da-eval embeddings --results results/rtdetr/source_AFO --wandb-project my_project
uv run da-eval embeddings --artifact embeddings-rtdetr-C:latest --wandb-project my_project
```

- **Projections:** PCA, t-SNE, UMAP and PaCMAP (`--methods` to choose) of source train, source
  test and every target test, on a balanced subset (same number of images per group; `--no-balance`,
  `--max-per-group`, `--seed`). Saved as PNG and logged as `embeddings_<method>/<level>`
  (one W&B section per projection, e.g. `embeddings_tsne` with the panels `all`, `P3`, `P4`, `P5`).
- **Distribution metrics (pymdma):** improved precision, improved recall, density and coverage of
  the source test and each target test against the **source train** embeddings (full sets; `--k`,
  `--metric`). Saved to `embedding_metrics.csv`, with per-image scores in
  `instance_scores_<dataset>_<split>.csv`; logged as `<dataset>/pymdma/<metric>` and
  "Table 3: Embedding distribution metrics". The table also has the MMD² of each target test
  against the source test (`<dataset>/mmd2`), computed as in the evaluation.
- **Levels:** the default embedding concatenates the pooled encoder outputs, [P3 | P4 | P5]
  (256 columns each for RT-DETRv4 and D-FINE). `--levels all P3 P4 P5` runs the analysis on the
  full vector and on each level separately, with no re-evaluation. Each level goes to
  `embedding_analysis/<level>/` and its metrics are logged with `/<level>` added to the keys
  (`<dataset>/pymdma/P3/density`, `<dataset>/mmd2/P3`); `all` keeps the plain metric keys.
  Besides Table 3 (one row per split and level), each metric gets its own table with one column
  per level, "Table 3a: Improved precision by level" … "Table 3e: MMD2 vs source test by level",
  also saved as `embedding_metrics_<metric>.csv`. The level layout is recorded in `embeddings_manifest.json` (`embedding_levels`);
  older results are assumed to be the default [P3 | P4 | P5] encoder embedding.
- **W&B:** a separate run (`job_type=embedding-analysis`) in the same group as the evaluation run;
  `--wandb-run-id <id>` adds the results to an existing run instead (e.g. the evaluation run, to
  have mAP and the embedding metrics side by side). `--artifact` records which evaluation produced
  the embeddings.
- Outputs go to `<results>/embedding_analysis/` (`--output-dir` to change). Results from before
  manifests existed are supported (groups are inferred from the file names).

For standalone embedding visualizations, `EmbeddingEvaluator.plot_embeddings`
uses the same Matplotlib-based image workflow as the shared t-SNE helper. It
returns a Matplotlib figure for local use, or can log a `wandb.Image` directly.
UMAP, PaCMAP and the pymdma metrics need the optional analysis extra (not required for
training or evaluation): `pip install "da-eval[analysis]"`, or in a model repository
`uv add "da-eval[analysis] @ git+https://github.com/JosmarSuarezUA/da-eval@<tag>"`.

```python
from da_eval.embedding_evaluation import EmbeddingEvaluator

evaluator = EmbeddingEvaluator(embedding_paths)
coordinates = evaluator.run_tsne(balanced=False)
image = evaluator.plot_embeddings(
    coordinates,
    balanced=False,
    wandb_run=run,
    wandb_key="embeddings/tsne",
)
```

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

Then register it in `da_eval/adapters/__init__.py`. `da_eval.adapters.common` has the shared
category mapping and the embedding pooling (`pool_features`, `feature_levels` for the P3/P4/P5
layout in the manifest). The command-line options an adapter receives are the arguments of its
constructor; any other option is rejected with a clear message.

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
    common.py           category mapping, embedding hook / pooling / level layout
    yaml_engine.py      RT-DETRv4 and D-FINE (shared YAMLConfig engine)
    ultralytics.py      Ultralytics YOLO checkpoints (yolo extra)
  embedding_evaluation.py  EmbeddingEvaluator: projections and pymdma metrics (analysis extra)
  embedding_analysis.py    da-eval embeddings command
  cli.py                da-eval command (evaluation, check, embeddings)
  train.py              da-train command
datasets.example.yaml   template for your datasets file
```
