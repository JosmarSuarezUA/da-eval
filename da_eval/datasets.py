"""
da_eval.datasets
================
User-defined datasets: one YAML file drives both training (``da-train``) and
evaluation (``da-eval``). Each entry is either COCO (annotation JSON + image
folder per split) or a YOLOv5 ``data.yaml``, which is converted to COCO JSON
and cached, since the detectors and pycocotools only read COCO.

Example::

    sds:                                   # key used in commands (--source sds)
      label: SeaDronesSee                  # optional display name
      train: {img_folder: /data/sds/images/train, ann_file: /data/sds/train.json}
      val:   {img_folder: /data/sds/images/val,   ann_file: /data/sds/val.json}
      test:  {img_folder: /data/sds/images/test,  ann_file: /data/sds/test.json}
      iou: 0.20                            # optional evaluation settings
      min_score: 0.001

    afo:
      label: AFO
      yolo: /data/afo/data.yaml            # YOLOv5 format: splits/names read from it

Paths are absolute or relative to the YAML file. Which splits are needed
depends on the use: training needs train + val, a source needs val + test, a
target needs test; train is optional for evaluation (t-SNE only).

Sections
--------
1.  Loading the user YAML (load_dataset_configs)
2.  YOLOv5 -> COCO conversion (cached)
3.  Training helpers (zero-based category ids)
4.  Validation (check_datasets_file)
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "val", "test")
DATASET_KEYS = {"label", "iou", "min_score", "yolo", "splits", *SPLITS}
# Same image suffixes as Ultralytics/YOLOv5.
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}
_CONVERTER_VERSION = "1"


# ---------------------------------------------------------------------------
# 1. Loading the User YAML
# ---------------------------------------------------------------------------

def _resolve(path: str | Path, base: str | Path) -> str:
    """Absolute path; relative paths are taken relative to ``base``. Symlinks are kept."""
    p = os.path.expanduser(os.path.expandvars(str(path)))
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(base, p))


def load_dataset_configs(path: str | Path, verbose: bool = True) -> dict:
    """Load the user's datasets YAML into the internal per-dataset config.

    Returns ``{name: {"label", "format", ["iou"], ["min_score"],
    "splits": {split: {"img_folder", "ann_file"}}}}`` with absolute paths.
    YOLO entries are converted to (cached) COCO JSON here.
    """
    path = os.path.abspath(os.path.expanduser(str(path)))
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must map dataset names to dataset definitions")
    return {
        str(name): parse_dataset(str(name), cfg, os.path.dirname(path), verbose=verbose)
        for name, cfg in raw.items()
    }


def parse_dataset(name: str, cfg: Any, base: str, verbose: bool = True) -> dict:
    """Validate and normalise one dataset entry of the user YAML."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Dataset '{name}': expected a mapping, got {type(cfg).__name__}")
    unknown = set(cfg) - DATASET_KEYS
    if unknown:
        raise ValueError(
            f"Dataset '{name}': unknown keys {sorted(unknown)} "
            f"(allowed: label, train, val, test, yolo, iou, min_score)"
        )

    out: dict[str, Any] = {k: cfg[k] for k in ("label", "iou", "min_score") if k in cfg}

    if "yolo" in cfg:
        if any(k in cfg for k in (*SPLITS, "splits")):
            raise ValueError(f"Dataset '{name}': use either 'yolo' or train/val/test, not both")
        out["format"] = "yolo"
        out["yolo"] = _resolve(cfg["yolo"], base)
        out["splits"] = yolo_to_coco_splits(out["yolo"], name, verbose=verbose)
        return out

    # 'splits:' nesting is still accepted for older files.
    raw_splits = cfg.get("splits", {s: cfg[s] for s in SPLITS if s in cfg})
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError(f"Dataset '{name}': define at least one of train/val/test, or 'yolo'")
    splits = {}
    for split, spec in raw_splits.items():
        if split not in SPLITS:
            raise ValueError(f"Dataset '{name}': unknown split '{split}' (allowed: train, val, test)")
        if not isinstance(spec, dict) or "img_folder" not in spec or "ann_file" not in spec:
            raise ValueError(f"Dataset '{name}', split '{split}': needs 'img_folder' and 'ann_file'")
        splits[split] = {
            "img_folder": _resolve(spec["img_folder"], base),
            "ann_file": _resolve(spec["ann_file"], base),
        }
    out["format"] = "coco"
    out["splits"] = splits
    return out


def read_categories(ann_file: str | Path) -> list[tuple[int, str]]:
    """Categories of a COCO file as ``[(id, name), ...]`` sorted by id."""
    with open(ann_file) as fh:
        cats = json.load(fh).get("categories", [])
    return sorted((int(c["id"]), str(c.get("name", c["id"]))) for c in cats)


# ---------------------------------------------------------------------------
# 2. YOLOv5 -> COCO Conversion (cached)
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Where converted YOLO splits are stored (``DA_EVAL_CACHE`` or ``~/.cache/da-eval``)."""
    return Path(os.path.expanduser(os.environ.get("DA_EVAL_CACHE", "~/.cache/da-eval")))


def read_yolo_data(data_yaml: str) -> dict:
    """Read a YOLOv5 ``data.yaml``: dataset root, class names and split entries."""
    with open(data_yaml) as fh:
        data = yaml.safe_load(fh) or {}
    yaml_dir = os.path.dirname(data_yaml)
    root = _resolve(data["path"], yaml_dir) if data.get("path") else yaml_dir

    names = data.get("names")
    if isinstance(names, dict):
        keys = sorted(int(k) for k in names)
        if keys != list(range(len(keys))):
            raise ValueError(f"{data_yaml}: 'names' keys must be 0..N-1, got {keys}")
        names = [str(names[k]) for k in keys]
    elif isinstance(names, list):
        names = [str(n) for n in names]
    elif data.get("nc") is not None:
        names = [str(i) for i in range(int(data["nc"]))]
    else:
        raise ValueError(f"{data_yaml}: needs 'names' (or 'nc')")
    if data.get("nc") is not None and int(data["nc"]) != len(names):
        raise ValueError(f"{data_yaml}: nc={data['nc']} but {len(names)} names")

    splits = {s: data[s] for s in SPLITS if data.get(s)}
    if not splits:
        raise ValueError(f"{data_yaml}: no train/val/test entries")
    return {"root": root, "names": names, "splits": splits}


def list_yolo_images(entry: str | list, root: str) -> list[str]:
    """Images of one split entry: a folder (recursive), a .txt list, an image, or a list of these.

    Relative entries are resolved against the dataset root; relative lines in a
    .txt list against the folder of that .txt file.
    """
    files: list[str] = []
    for item in entry if isinstance(entry, list) else [entry]:
        p = _resolve(item, root)
        if os.path.isdir(p):
            files += sorted(
                str(f) for f in Path(p).rglob("*")
                if f.is_file() and f.suffix[1:].lower() in IMG_FORMATS
            )
        elif os.path.isfile(p) and p.endswith(".txt"):
            with open(p) as fh:
                files += [_resolve(ln.strip(), os.path.dirname(p)) for ln in fh if ln.strip()]
        elif os.path.isfile(p) and p.rsplit(".", 1)[-1].lower() in IMG_FORMATS:
            files.append(p)
        else:
            raise FileNotFoundError(f"YOLO split entry not found or not an image folder/.txt list: {p}")
    files = list(dict.fromkeys(files))  # de-duplicate, keep order
    if not files:
        raise ValueError(f"No images found for YOLO split entry {entry!r} (root {root})")
    return files


def yolo_label_path(image_path: str) -> str:
    """YOLOv5 convention: ``.../images/x.jpg`` -> ``.../labels/x.txt``."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(image_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def _fingerprint(images: list[str], names: list[str]) -> str:
    h = hashlib.sha1(f"{_CONVERTER_VERSION}|{json.dumps(names)}".encode())
    for f in images:
        st = os.stat(f)
        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode())
        try:
            st = os.stat(yolo_label_path(f))
            h.update(f"|{st.st_size}|{st.st_mtime_ns}\n".encode())
        except FileNotFoundError:
            h.update(b"|-\n")
    return h.hexdigest()


def yolo_split_to_coco(images: list[str], names: list[str]) -> tuple[dict, str, dict]:
    """Convert YOLO labels of ``images`` to a COCO dict.

    Category ids are the YOLO class indices (0..N-1); image ``file_name`` is
    relative to the returned image folder (the images' common directory).
    Returns ``(coco_dict, img_folder, stats)``.
    """
    from PIL import Image

    img_folder = os.path.commonpath([os.path.dirname(f) for f in images])
    coco: dict[str, list] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
    }
    stats = Counter()
    for img_id, f in enumerate(images, start=1):
        with Image.open(f) as im:
            w, h = im.size
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):  # rotated by EXIF orientation
                stats["exif_rotated"] += 1
        coco["images"].append({"id": img_id, "file_name": os.path.relpath(f, img_folder), "width": w, "height": h})

        label_file = yolo_label_path(f)
        if not os.path.exists(label_file):
            stats["images_without_label_file"] += 1
            continue
        with open(label_file) as fh:
            for line_no, line in enumerate(fh, start=1):
                v = line.split()
                if not v:
                    continue
                where = f"{label_file}:{line_no}"
                cls = int(float(v[0]))
                if not 0 <= cls < len(names):
                    raise ValueError(f"{where}: class {cls} outside 0..{len(names) - 1}")
                vals = [float(x) for x in v[1:]]
                if len(vals) == 4:
                    cx, cy, bw, bh = vals
                    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon (segmentation label)
                    xs, ys = vals[0::2], vals[1::2]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    stats["polygons_converted_to_boxes"] += 1
                else:
                    raise ValueError(f"{where}: expected 'class cx cy w h' or 'class x1 y1 x2 y2 ...'")
                if min(x0, y0) < 0 or max(x1, y1) > 1:
                    stats["boxes_clipped_to_image"] += 1
                x0, x1 = max(0.0, x0) * w, min(1.0, x1) * w
                y0, y1 = max(0.0, y0) * h, min(1.0, y1) * h
                if x1 <= x0 or y1 <= y0:
                    stats["empty_boxes_dropped"] += 1
                    continue
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                })
    return coco, img_folder, dict(stats)


def _write_json_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def yolo_to_coco_splits(data_yaml: str, name: str = "", verbose: bool = True) -> dict:
    """COCO ``{"img_folder", "ann_file", "yolo_stats"}`` per split of a YOLOv5 dataset.

    Each split is converted once and cached; it is rebuilt only when the image
    list, an image or a label file changes (sizes and modification times).
    """
    info = read_yolo_data(data_yaml)
    splits = {}
    for split, entry in info["splits"].items():
        images = list_yolo_images(entry, info["root"])
        fp = _fingerprint(images, info["names"])
        key = hashlib.sha1(f"{data_yaml}|{split}".encode()).hexdigest()[:16]
        ann_file = cache_dir() / f"yolo_{key}_{split}.json"
        meta_file = ann_file.with_suffix(".meta.json")

        meta = None
        if ann_file.exists() and meta_file.exists():
            with open(meta_file) as fh:
                meta = json.load(fh)
            if meta.get("fingerprint") != fp:
                meta = None
        if meta is None:
            if verbose:
                print(f"[*] Converting YOLO split '{split}' of '{name or data_yaml}' to COCO ({len(images)} images)")
            coco, img_folder, stats = yolo_split_to_coco(images, info["names"])
            _write_json_atomic(coco, ann_file)
            meta = {"fingerprint": fp, "img_folder": img_folder, "stats": stats, "data_yaml": data_yaml}
            _write_json_atomic(meta, meta_file)

        splits[split] = {"img_folder": meta["img_folder"], "ann_file": str(ann_file), "yolo_stats": meta["stats"]}
    return splits


# ---------------------------------------------------------------------------
# 3. Training Helpers
# ---------------------------------------------------------------------------

def write_zero_based_coco(ann_file: str | Path, out_file: str | Path) -> list[str]:
    """Copy a COCO file with category ids remapped to 0..K-1 (sorted by original id).

    Detectors built on the RT-DETR engine use ``category_id`` directly as the
    class index, so ids must be 0..num_classes-1. Returns the class names in
    index order. Evaluation maps model index i back to the i-th sorted id.
    """
    with open(ann_file) as fh:
        data = json.load(fh)
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    if not cats:
        raise ValueError(f"{ann_file}: no categories")
    mapping = {c["id"]: i for i, c in enumerate(cats)}
    data["categories"] = [{**c, "id": mapping[c["id"]]} for c in cats]
    for ann in data.get("annotations", []):
        if ann["category_id"] not in mapping:
            raise ValueError(f"{ann_file}: annotation {ann.get('id')} has unknown category_id {ann['category_id']}")
        ann["category_id"] = mapping[ann["category_id"]]
    _write_json_atomic(data, Path(out_file))
    return [str(c.get("name", c["id"])) for c in cats]


# ---------------------------------------------------------------------------
# 4. Validation (check_datasets_file)
# ---------------------------------------------------------------------------

def check_split(split: dict, max_examples: int = 3) -> tuple[dict, list[str], list[str]]:
    """Validate one COCO split. Returns ``(summary, errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []
    if not os.path.isfile(split["ann_file"]):
        return {}, [f"annotation file not found: {split['ann_file']}"], []
    if not os.path.isdir(split["img_folder"]):
        errors.append(f"image folder not found: {split['img_folder']}")
    try:
        with open(split["ann_file"]) as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return {}, [f"invalid JSON in {split['ann_file']}: {exc}"], []
    missing_keys = [k for k in ("images", "annotations", "categories") if k not in data]
    if missing_keys:
        return {}, [f"not a COCO detection file (missing {missing_keys}): {split['ann_file']}"], []

    images, anns = data["images"], data["annotations"]
    cat_ids = {c["id"] for c in data["categories"]}
    img_ids = [im["id"] for im in images]
    if len(set(img_ids)) != len(img_ids):
        errors.append("duplicate image ids")
    sizes = {im["id"]: (im.get("width"), im.get("height")) for im in images}

    missing_imgs = [im["file_name"] for im in images
                    if not os.path.isfile(os.path.join(split["img_folder"], im["file_name"]))]
    if missing_imgs:
        errors.append(f"{len(missing_imgs)}/{len(images)} images not found under the image folder, "
                      f"e.g. {missing_imgs[:max_examples]}")

    bad_cat = [a["id"] for a in anns if a.get("category_id") not in cat_ids]
    if bad_cat:
        errors.append(f"{len(bad_cat)} annotations use a category_id missing from 'categories'")
    bad_img = [a["id"] for a in anns if a.get("image_id") not in sizes]
    if bad_img:
        errors.append(f"{len(bad_img)} annotations refer to unknown image ids")
    degenerate = [a["id"] for a in anns if len(a.get("bbox", [])) != 4 or a["bbox"][2] <= 0 or a["bbox"][3] <= 0]
    if degenerate:
        warnings.append(f"{len(degenerate)} boxes with zero/negative width or height")
    outside = 0
    for a in anns:
        w, h = sizes.get(a.get("image_id"), (None, None))
        if w and h and len(a.get("bbox", [])) == 4:
            x, y, bw, bh = a["bbox"]
            if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
                outside += 1
    if outside:
        warnings.append(f"{outside} boxes extend outside their image")

    for key, count in (split.get("yolo_stats") or {}).items():
        if key == "images_without_label_file":
            continue  # normal in YOLO: background images
        warnings.append(f"YOLO conversion: {count} {key.replace('_', ' ')}")

    summary = {
        "images": len(images),
        "annotations": len(anns),
        "images_without_objects": len(set(img_ids) - {a.get("image_id") for a in anns}),
        "categories": sorted((c["id"], c.get("name", "")) for c in data["categories"]),
    }
    return summary, errors, warnings


def check_datasets_file(path: str | Path) -> int:
    """Print a validation report for a datasets YAML. Returns the number of errors."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        print(f"[✗] Cannot read {path}: {exc}")
        return 1
    if not isinstance(raw, dict) or not raw:
        print(f"[✗] {path} must map dataset names to dataset definitions")
        return 1

    n_errors = 0
    class_names: dict[str, list[str]] = {}
    print(f"Checking {path}\n")
    for name, cfg in raw.items():
        name = str(name)
        try:
            ds = parse_dataset(name, cfg, os.path.dirname(path), verbose=True)
        except Exception as exc:  # report and continue with the other datasets
            print(f"[✗] {name}: {exc}\n")
            n_errors += 1
            continue
        print(f"=== {name} ({ds.get('label', name)}, {ds['format']})")
        for split in SPLITS:
            if split not in ds["splits"]:
                continue
            summary, errors, warnings = check_split(ds["splits"][split])
            if summary:
                cats = ", ".join(f"{i}:{n}" for i, n in summary["categories"])
                print(f"  {split:5s}: {summary['images']} images, {summary['annotations']} boxes, "
                      f"{summary['images_without_objects']} images without objects | classes [{cats}]")
                class_names.setdefault(name, [n for _, n in summary["categories"]])
                if [n for _, n in summary["categories"]] != class_names[name]:
                    errors.append("class list differs from this dataset's other splits")
            for e in errors:
                print(f"  [✗] {split}: {e}")
            for w in warnings:
                print(f"  [!] {split}: {w}")
            n_errors += len(errors)
        has = set(ds["splits"])
        uses = [use for use, req in (("training", {"train", "val"}), ("evaluation source", {"val", "test"}),
                                     ("evaluation target", {"test"})) if req <= has]
        print(f"  usable for: {', '.join(uses) if uses else 'nothing (add splits)'}\n")

    distinct = {tuple(v) for v in class_names.values()}
    if len(distinct) > 1:
        print("[!] Class lists differ between datasets; predictions are matched to classes by sorted "
              "category id, so cross-dataset results are only meaningful if the order matches:")
        for name, names in class_names.items():
            print(f"      {name}: {names}")
    print(f"{'[✓] No errors found' if n_errors == 0 else f'[✗] {n_errors} error(s) found'}")
    return n_errors
