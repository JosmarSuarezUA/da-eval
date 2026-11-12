"""
da_eval.adapters.ultralytics
============================
Adapter for Ultralytics YOLO detection checkpoints (``best.pt``), e.g. YOLO26.
Needs the yolo extra: ``pip install "da-eval[yolo]"``.

The checkpoint holds the whole model, so no config file or model repository is
needed. Images are preprocessed by Ultralytics itself (square letterbox to the
checkpoint's training ``imgsz``), and predictions + embeddings come from the
same forward pass, as for the other adapters.

Embeddings: by default the inputs of the ``Detect`` head, i.e. the neck outputs
P3/P4/P5 (the counterpart of the RT-DETR encoder output), each average-pooled and
concatenated. Any other module can be hooked by name (``model.<index>``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from da_eval.adapters.common import feature_levels, get_label_to_category_map, pool_features
from da_eval.pipeline import DetectorAdapter, SplitOutput

NECK = "neck"   # embedding_module value for the Detect-head inputs (P3, P4, P5)


class UltralyticsAdapter(DetectorAdapter):
    """Inference with an Ultralytics YOLO detection checkpoint.

    Parameters
    ----------
    device           : torch device, e.g. "cuda:0" or "cpu" (default: cuda:0 if available).
    embedding_module : "neck" (default) = Detect-head inputs P3/P4/P5; or the name of any
                       module of the model, e.g. "model.10", hooked on its output.
    batch_size       : images per forward pass.
    num_workers      : threads reading images.
    imgsz            : inference size; default = the checkpoint's training ``imgsz``.
    max_det          : boxes kept per image (Ultralytics default 300, as RT-DETR's 300 queries).
    """

    name = "YOLO"
    slug = "yolo"

    def __init__(
        self,
        device: str | torch.device | None = None,
        embedding_module: str = NECK,
        batch_size: int = 8,
        num_workers: int = 4,
        imgsz: int | None = None,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.embedding_module = embedding_module
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.imgsz_override = imgsz
        self.max_det = max_det

        self.yolo = None
        self.imgsz: int | None = None
        self.config_path: str | None = None
        self._features: list = []         # hooked features of each forward pass
        self._input_widths: list[int] = []
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    def _load(self, checkpoint_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('Ultralytics is needed for YOLO models: pip install "da-eval[yolo]"') from exc

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self.embedding_levels = None

        self.yolo = YOLO(checkpoint_path, task="detect")
        model = self.yolo.model
        num_classes = len(model.names)
        if self.num_classes is not None and num_classes != self.num_classes:
            raise ValueError(
                f"{checkpoint_path} predicts {num_classes} classes {dict(model.names)}, but its source "
                f"dataset has {self.num_classes}. Class index i must be the i-th category (sorted by id)."
            )

        train_imgsz = (getattr(model, "args", None) or {}).get("imgsz", 640)
        if isinstance(train_imgsz, (list, tuple)):
            train_imgsz = max(train_imgsz)
        self.imgsz = int(self.imgsz_override or train_imgsz)
        # Architecture file the checkpoint was built from (e.g. yolo26n.yaml): the counterpart
        # of the RT-DETR / D-FINE model config, reported as config_path.
        self.config_path = str(model.yaml.get("yaml_file", "")) or None
        stem = Path(self.config_path or "yolo").stem
        self.name = f"YOLO{stem[4:]}" if stem.lower().startswith("yolo") else stem   # yolo26n -> YOLO26n

        def store_input(module, args):
            self._input_widths.append(int(args[0].shape[-1]))

        if self.embedding_module == NECK:
            def store(module, args):   # copy the list: older Detect heads overwrite its items in place
                self._features.append(list(args[0]))
            hook = model.model[-1].register_forward_pre_hook(store)
        else:
            modules = dict(model.named_modules())
            if self.embedding_module not in modules:
                raise ValueError(
                    f"Module '{self.embedding_module}' not found. Use '{NECK}' or one of: "
                    f"{[n for n in modules if n.count('.') == 1][:30]}"
                )
            hook = modules[self.embedding_module].register_forward_hook(
                lambda module, args, out: self._features.append(out))
        self._hook_handles = [hook, model.register_forward_pre_hook(store_input)]

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "config_path": self.config_path, "embedding_module": self.embedding_module}

    def preprocessing(self) -> dict[str, Any]:
        """Square letterbox (aspect ratio kept, grey padding) to ``imgsz``."""
        return {"input_size": [self.imgsz, self.imgsz] if self.imgsz else None, "resize": "letterbox"}

    # ------------------------------------------------------------------
    def run_split(
        self,
        split: dict,
        *,
        min_score: float = 0.001,
        with_predictions: bool = True,
        with_embeddings: bool = False,
        verbose: bool = True,
    ) -> SplitOutput:
        if self.yolo is None:
            raise RuntimeError("Call load(checkpoint_path) before running inference.")
        from pycocotools.coco import COCO
        from ultralytics.utils.patches import imread

        coco_gt = COCO(split["ann_file"])
        label_to_cat = get_label_to_category_map(coco_gt)
        image_ids = sorted(coco_gt.imgs.keys())   # same order as the RT-DETR / D-FINE dataloaders
        file_names = [coco_gt.imgs[i]["file_name"] for i in image_ids]
        img_folder = Path(split["img_folder"])

        def read(file_name: str) -> np.ndarray:
            image = imread(str(img_folder / file_name))
            if image is None:
                raise FileNotFoundError(f"Could not read image {img_folder / file_name}")
            return image

        batches = range(0, len(image_ids), self.batch_size)
        try:
            from tqdm import tqdm
            batches = tqdm(batches, desc="Inference", leave=False)
        except ImportError:
            pass

        predictions: list[dict] = []
        emb_list: list[np.ndarray] = []
        with ThreadPoolExecutor(max(1, self.num_workers)) as pool:
            for start in batches:
                ids = image_ids[start:start + self.batch_size]
                images = list(pool.map(read, file_names[start:start + self.batch_size]))

                self._features.clear()
                self._input_widths.clear()
                results = self.yolo.predict(
                    images, imgsz=self.imgsz, conf=min_score, max_det=self.max_det, batch=len(images),
                    device=self.device, half=False, rect=False, save=False, verbose=False,
                )
                # The last forward pass is this batch (a warm-up pass may run before it).
                features, input_width = self._features[-1], self._input_widths[-1]
                pooled = pool_features(features)
                if pooled.shape[0] != len(images):
                    raise RuntimeError(f"Expected one forward pass of {len(images)} images, got batch {pooled.shape[0]}.")

                if with_predictions:
                    for img_id, res in zip(ids, results):
                        predictions.extend(self._to_coco(img_id, res, label_to_cat))
                if with_embeddings:
                    if self.embedding_levels is None:
                        self.embedding_levels = feature_levels(features, input_width)
                    emb_list.extend(pooled.detach().float().cpu().numpy())

        embeddings = np.stack(emb_list) if with_embeddings and emb_list else None
        if verbose:
            if with_predictions:
                print(f"[*] Total predictions collected: {len(predictions)}")
            if embeddings is not None:
                print(f"[*] Embeddings extracted (shape: {embeddings.shape})")

        return SplitOutput(
            predictions=predictions,
            coco_gt=coco_gt,
            image_ids=image_ids,
            file_names=file_names,
            embeddings=embeddings,
        )

    @staticmethod
    def _to_coco(img_id: int, res: Any, label_to_cat: dict[int, int]) -> list[dict]:
        boxes = res.boxes.xyxy.cpu().numpy()     # original image pixels
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(int)
        return [
            {
                "image_id": img_id,
                "category_id": label_to_cat[int(lbl)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
            for (x1, y1, x2, y2), score, lbl in zip(boxes, scores, labels)
        ]
