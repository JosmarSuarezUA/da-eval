"""
da_eval.cli
===========
``da-eval`` command: evaluate every source checkpoint of one model against the
test split of every dataset in the user's datasets YAML.

Run it from inside the model repository, e.g. in RT-DETRv4::

    uv run da-eval --model rtdetrv4 \\
        --config configs/rtv4/rtv4_hgnetv2_s_coco.yml \\
        --datasets ~/my_datasets.yaml \\
        --checkpoint sds=output/rtv4_hgnetv2_s_coco_sds/best_stg1.pth \\
        --checkpoint afo=output/rtv4_hgnetv2_s_coco_afo/best_stg1.pth

Validate a datasets YAML (paths, COCO files, images, classes)::

    uv run da-eval check --datasets ~/my_datasets.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

from da_eval.adapters import ADAPTERS, get_adapter_class
from da_eval.datasets import check_datasets_file, load_dataset_configs
from da_eval.pipeline import print_results_summary, run_all_sources


def parse_updates(items: list[str] | None) -> dict:
    """``["a.b=1", "c=x"]`` -> ``{"a": {"b": 1}, "c": "x"}`` (values parsed as YAML), like train.py -u."""
    import yaml

    out: dict = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"-u expects key=value, got '{item}'")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return out


def _parse_checkpoints(items: list[str], dataset_configs: dict) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for item in items:
        source, sep, path = item.partition("=")
        if not sep or not source or not path:
            raise SystemExit(f"--checkpoint expects SOURCE=PATH, got '{item}'")
        if source not in dataset_configs:
            raise SystemExit(f"Unknown source '{source}'. Datasets: {sorted(dataset_configs)}")
        checkpoints[source] = path
    return checkpoints


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="da-eval",
        description="Cross-domain evaluation: each source checkpoint vs. every dataset's test split. "
                    "Use 'da-eval check --datasets FILE' to validate a datasets YAML.",
    )
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS), help="Detector adapter")
    p.add_argument("--config", "-c", required=True, help="Model YAML config")
    p.add_argument("--datasets", required=True, help="Datasets YAML (see datasets.example.yaml)")
    p.add_argument("--checkpoint", "-r", action="append", required=True, metavar="SOURCE=PATH",
                   help="Checkpoint trained on SOURCE (repeat for each source)")
    p.add_argument("--result-root", default=None, help="Output directory (default: results/<model>)")
    p.add_argument("--repo-root", default=".", help="Model repository root")
    p.add_argument("--device", default=None, help="Torch device (default: cuda:0 if available)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--embedding-module", default="encoder", help="Module to hook for embeddings")
    p.add_argument("--remap-mscoco", action="store_true", help="Standard MS COCO 80-class mapping")
    p.add_argument("--no-embeddings", action="store_true", help="Skip embeddings, t-SNE and MMD")
    p.add_argument("--wandb-project", default=None, help="Optional W&B project name")
    p.add_argument("--wandb-entity", default=None, help="Optional W&B entity/team name")
    p.add_argument("-u", "--update", nargs="+", metavar="KEY=VALUE",
                   help="Override model config keys, like train.py -u (e.g. HGNetv2.pretrained=False)")
    return p


def check_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="da-eval check", description="Validate a datasets YAML.")
    p.add_argument("--datasets", required=True, help="Datasets YAML to check")
    args = p.parse_args(argv)
    raise SystemExit(1 if check_datasets_file(args.datasets) else 0)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["check"]:
        return check_main(argv[1:])
    args = build_parser().parse_args(argv)
    dataset_configs = load_dataset_configs(args.datasets)
    checkpoints = _parse_checkpoints(args.checkpoint, dataset_configs)

    adapter = get_adapter_class(args.model)(
        config_path=args.config,
        repo_root=args.repo_root,
        device=args.device,
        embedding_module=args.embedding_module,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remap_mscoco=args.remap_mscoco,
        yaml_overrides=parse_updates(args.update),
    )
    result_root = args.result_root or os.path.join("results", adapter.slug)

    all_results = run_all_sources(
        adapter,
        checkpoints=checkpoints,
        dataset_configs=dataset_configs,
        result_root=result_root,
        extract_embeddings_flag=not args.no_embeddings,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    for source, results in all_results.items():
        print(f"\n### Source {source} ({dataset_configs[source].get('label', source)})")
        print_results_summary(results)
    print(f"[✓] All evaluation outputs saved to: {os.path.abspath(result_root)}")


if __name__ == "__main__":
    main()
