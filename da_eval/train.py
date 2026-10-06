"""
da_eval.train
=============
``da-train`` command: train an RT-DETR-engine detector (RT-DETRv4, D-FINE) on
one dataset of the user's datasets YAML, using the repository's own
``train.py`` and any of its model configs.

Run it from inside the model repository::

    uv run da-train --config configs/rtv4/rtv4_hgnetv2_s_coco.yml \\
        --datasets ~/my_datasets.yaml --source sds --gpus 0,1 \\
        -- --use-amp --seed 0 -t weights.pth

What it does:
  1. Writes zero-based copies of the source's train/val annotations (category
     ids -> 0..K-1, sorted by id) into ``<output-dir>/annotations/``, because
     these repos use ``category_id`` directly as the class index.
  2. Calls ``torchrun train.py -c <config>`` overriding only the dataset paths,
     ``num_classes``, ``remap_mscoco_category`` and the output directory, so
     the model, schedule and augmentations of the config are kept.
Everything after ``--`` is passed to ``train.py`` unchanged; extra ``-u
key=value`` overrides there are applied after the ones above.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from da_eval.datasets import load_dataset_configs, write_zero_based_coco
from da_eval.pipeline import get_split


def split_update_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Separate ``-u/--update key=value ...`` values from the other train.py arguments."""
    updates: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-u", "--update"):
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                updates.append(args[i])
                i += 1
        else:
            rest.append(args[i])
            i += 1
    return updates, rest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="da-train",
        description="Train on one dataset of a datasets YAML (run inside the RT-DETRv4 or D-FINE repo). "
                    "Arguments after '--' go to train.py.",
    )
    p.add_argument("--config", "-c", required=True, help="Model YAML config of the repository")
    p.add_argument("--datasets", required=True, help="Datasets YAML (see datasets.example.yaml)")
    p.add_argument("--source", required=True, help="Dataset key to train on (needs train and val splits)")
    p.add_argument("--gpus", default="0", help="Comma-separated GPU ids (default: 0)")
    p.add_argument("--master-port", type=int, default=7777, help="torchrun master port")
    p.add_argument("--output-dir", default=None,
                   help="Run directory (default: output/<config name>_<source>)")
    p.add_argument("--dry-run", action="store_true", help="Prepare annotations and print the command only")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        sep = argv.index("--")
        own, passthrough = argv[:sep], argv[sep + 1:]
    else:
        own, passthrough = argv, []
    args = build_parser().parse_args(own)

    if not os.path.isfile("train.py"):
        raise SystemExit("da-train must be run from the model repository root (train.py not found).")

    dataset_configs = load_dataset_configs(args.datasets)
    if args.source not in dataset_configs:
        raise SystemExit(f"Unknown source '{args.source}'. Datasets: {sorted(dataset_configs)}")
    ds = dataset_configs[args.source]
    train_split = get_split(ds, "train", args.source)
    val_split = get_split(ds, "val", args.source)

    out_dir = Path(args.output_dir or Path("output") / f"{Path(args.config).stem}_{args.source}")
    ann_dir = (out_dir / "annotations").resolve()
    names = write_zero_based_coco(train_split["ann_file"], ann_dir / "train.json")
    val_names = write_zero_based_coco(val_split["ann_file"], ann_dir / "val.json")
    if names != val_names:
        raise SystemExit(f"'{args.source}': train classes {names} differ from val classes {val_names}")

    updates = [
        f"num_classes={len(names)}",
        "remap_mscoco_category=False",
        f"train_dataloader.dataset.img_folder={train_split['img_folder']}",
        f"train_dataloader.dataset.ann_file={ann_dir / 'train.json'}",
        f"val_dataloader.dataset.img_folder={val_split['img_folder']}",
        f"val_dataloader.dataset.ann_file={ann_dir / 'val.json'}",
    ]
    user_updates, rest = split_update_args(passthrough)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--master_port={args.master_port}", f"--nproc_per_node={len(gpus)}",
        "train.py", "-c", args.config, "--output-dir", str(out_dir),
        *rest, "-u", *updates, *user_updates,
    ]

    record = {
        "source": args.source,
        "label": ds.get("label", args.source),
        "datasets_yaml": os.path.abspath(args.datasets),
        "classes": names,
        "train": train_split,
        "val": val_split,
        "gpus": gpus,
        "command": cmd,
    }
    with open(out_dir / "da_train.json", "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"[*] Training on '{args.source}' ({len(names)} classes: {names}) -> {out_dir}")
    print(f"[*] CUDA_VISIBLE_DEVICES={','.join(gpus)} " + shlex.join(cmd))
    if args.dry_run:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
