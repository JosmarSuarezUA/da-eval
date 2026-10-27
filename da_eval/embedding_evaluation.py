"""
da_eval.embedding_evaluation
============================
Analysis of saved image embeddings: dimensionality-reduction plots (PCA,
t-SNE, UMAP, PaCMAP) and pymdma distribution metrics.

Needs the optional analysis dependencies::

    pip install "da-eval[analysis]"     # or: uv add "da-eval[analysis] @ git+..."

They are imported only when used, so training/evaluation environments don't need them.
"""

import importlib
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _require(module: str, package: str):
    """Import an optional analysis dependency, with an install hint if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"'{package}' is needed for embedding analysis. "
            f'Install the analysis extra: pip install "da-eval[analysis]"'
        ) from exc

class EmbeddingEvaluator:
    """
    Evaluate and visualize image-level embeddings.

    emb_path_dict:
        {
            "SeaDronesSee JP Train": "/path/embeddings.npy",
            "SeaDronesSee JP Test": "/path/embeddings.npy",
            "AFO Humans Test": "/path/embeddings.npy",
            ...
        }

    Each row corresponds to one image.
    Each column corresponds to one embedding feature.
    """

    def __init__(self, emb_path_dict, label_mapping=None):
        self.emb_path_dict = emb_path_dict
        self.embeddings_dict = {}

        self.label_mapping = (
            label_mapping
            if label_mapping is not None
            else {
                name: idx
                for idx, name in enumerate(emb_path_dict)
            }
        )

        if set(self.emb_path_dict) != set(self.label_mapping):
            raise ValueError(
                "label_mapping must contain exactly "
                "the embedding group names."
            )

        if len(set(self.label_mapping.values())) != len(
            self.label_mapping
        ):
            raise ValueError("Each group must have a unique label.")

        self.inverse_label_mapping = {
            v: k for k, v in self.label_mapping.items()
        }

        self.X = None
        self.y = None
        self.metadata = None

        # Selected sample indices relative to self.X
        self.selected_indices = None
        self.X_selected = None
        self.y_selected = None
        self.metadata_selected = None


        self.load_data()

    def load_data(self):
        """Load and concatenate all embedding groups."""
        pd = _require("pandas", "pandas")
        X_list = []
        y_list = []
        metadata_list = []
        expected_dim = None

        print("--- Loading Embeddings ---")

        for name, filepath in self.emb_path_dict.items():
            filepath = Path(filepath)

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Embedding file not found: {filepath}"
                )

            data = np.load(filepath)

            if data.ndim != 2:
                raise ValueError(
                    f"{name}: expected 2D array "
                    f"(n_images, embedding_dim), got {data.shape}"
                )

            if data.shape[0] == 0:
                raise ValueError(f"{name}: embedding array is empty.")

            if not np.issubdtype(data.dtype, np.number):
                raise ValueError(f"{name}: embeddings must be numeric.")

            if not np.isfinite(data).all():
                raise ValueError(
                    f"{name}: embeddings contain NaN or Inf."
                )

            if expected_dim is None:
                expected_dim = data.shape[1]
            elif data.shape[1] != expected_dim:
                raise ValueError(
                    f"{name}: embedding dimension "
                    f"{data.shape[1]} != {expected_dim}"
                )

            self.embeddings_dict[name] = data

            n_samples = data.shape[0]
            X_list.append(data)
            y_list.append(
                np.full(n_samples, self.label_mapping[name], dtype=int)
            )
            metadata_list.append(
                pd.DataFrame({
                    "group": name,
                    "sample_idx": np.arange(n_samples),
                })
            )

            print(
                f"{name}: {n_samples} images, "
                f"{data.shape[1]} features"
            )

        self.X = np.concatenate(X_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)
        self.metadata = pd.concat(metadata_list, ignore_index=True)
        self.metadata["global_idx"] = np.arange(len(self.metadata))

        print("\n--- Combined Embeddings ---")
        print(f"Samples: {self.X.shape[0]}")
        print(f"Features: {self.X.shape[1]}")
        print(f"Shape: {self.X.shape}")

    def balance(self, max_samples_per_group=None, random_state=42):
        """Create a balanced subset of groups."""
        rng = np.random.default_rng(random_state)
        group_ids = np.unique(self.y)

        group_indices = {
            group_id: np.flatnonzero(self.y == group_id)
            for group_id in group_ids
        }
        group_sizes = {
            group_id: len(indices)
            for group_id, indices in group_indices.items()
        }

        n_samples = min(group_sizes.values())

        if max_samples_per_group is not None:
            if max_samples_per_group < 1:
                raise ValueError(
                    "max_samples_per_group must be >= 1."
                )

            n_samples = min(n_samples, max_samples_per_group)

        selected = []

        for group_id in group_ids:
            indices = group_indices[group_id]
            chosen = rng.choice(
                indices,
                size=n_samples,
                replace=False,
            )
            selected.extend(chosen)

        selected = np.asarray(selected, dtype=int)
        rng.shuffle(selected)

        self.selected_indices = selected
        self.X_selected = self.X[selected]
        self.y_selected = self.y[selected]
        self.metadata_selected = (
            self.metadata.iloc[selected].reset_index(drop=True).copy()
        )

        print("\n--- Balanced Subset ---")
        print(f"Samples per group: {n_samples}")
        print(f"Total samples: {len(selected)}")
        print(self.metadata_selected["group"].value_counts())

        return self.X_selected, self.y_selected

    def _get_data(self, balanced=True):
        """Return balanced or full embeddings."""
        if balanced:
            if self.X_selected is None:
                raise ValueError(
                    "Run balance() first, or use balanced=False."
                )
            return self.X_selected, self.y_selected

        return self.X, self.y

    def run_pca(self, n_components=2, balanced=True, **kwargs):
        """Run PCA dimensionality reduction."""
        X, _ = self._get_data(balanced)
        reducer = PCA(n_components=n_components, **kwargs)
        return reducer.fit_transform(X)

    def run_tsne(
        self,
        n_components=2,
        perplexity=30,
        random_state=42,
        balanced=True,
        **kwargs,
    ):
        """Run t-SNE dimensionality reduction."""
        X, _ = self._get_data(balanced)

        if perplexity >= len(X):
            raise ValueError(
                "perplexity must be smaller than the number of samples."
            )

        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            **kwargs,
        )
        return reducer.fit_transform(X)

    def run_umap(self, n_components=2, random_state=42, balanced=True, **kwargs):
        """Run UMAP dimensionality reduction."""
        X, _ = self._get_data(balanced)
        umap = _require("umap", "umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(X)

    def run_pacmap(
        self,
        n_components=2,
        random_state=42,
        balanced=True,
        **kwargs,
    ):
        """Run PaCMAP dimensionality reduction."""
        X, _ = self._get_data(balanced)
        pacmap = _require("pacmap", "pacmap")
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
        return reducer.fit_transform(X)

    def plot_embeddings(
        self,
        coordinates,
        title="Embedding Visualization",
        balanced=True,
        cmap_name="Set1",
        wandb_run=None,
        wandb_key="embedding_visualization",
    ):
        """Plot reduced embeddings, optionally logging the plot to W&B.

        When ``wandb_run`` is omitted, the open Matplotlib figure is returned.
        When it is provided, the figure is wrapped in ``wandb.Image``, logged
        under ``wandb_key``, closed, and the image is returned.
        """
        coordinates = np.asarray(coordinates)
        _, y = self._get_data(balanced)

        if coordinates.ndim != 2 or coordinates.shape[1] < 2:
            raise ValueError("Coordinates must have at least two columns.")

        if coordinates.shape[0] != len(y):
            raise ValueError(
                "Coordinates and labels must have the same "
                "number of samples."
            )

        labels = [self.inverse_label_mapping[label] for label in y]
        original_order = list(self.emb_path_dict.keys())

        fig, ax = plt.subplots(figsize=(7, 7))
        cmap = mpl.colormaps[cmap_name]
        for index, group_name in enumerate(original_order):
            mask = np.asarray(labels) == group_name
            if not np.any(mask):
                continue
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                label=group_name,
                alpha=0.75,
                s=49,
                color=cmap.colors[index % len(cmap.colors)],
                linewidths=0.3,
                edgecolors="DarkSlateGrey",
            )

        ax.set(
            title=title,
            xlabel="Component 1",
            ylabel="Component 2",
        )
        ax.legend(title="Embedding Group")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()

        if wandb_run is not None:
            try:
                import wandb
            except ImportError as exc:
                plt.close(fig)
                raise ImportError(
                    "wandb is required when wandb_run is provided. "
                    "Install it with: pip install wandb"
                ) from exc

            image = wandb.Image(fig)
            wandb_run.log({wandb_key: image})
            plt.close(fig)
            return image

        return fig

    def compute_distribution_metrics(
        self,
        reference_features,
        target_features,
        k=5,
        metric="euclidean",
        n_workers=4,
    ):
        """Compare reference and target embedding distributions."""
        reference_features = np.asarray(reference_features)
        target_features = np.asarray(target_features)

        if reference_features.ndim != 2:
            raise ValueError("Reference embeddings must be 2D.")

        if target_features.ndim != 2:
            raise ValueError("Target embeddings must be 2D.")

        if reference_features.shape[1] != target_features.shape[1]:
            raise ValueError(
                "Reference and target dimensions must match."
            )

        if len(reference_features) <= k:
            raise ValueError("Reference set must contain more than k samples.")

        if len(target_features) <= k:
            raise ValueError("Target set must contain more than k samples.")

        if not np.isfinite(reference_features).all():
            raise ValueError("Reference embeddings contain NaN or Inf.")

        if not np.isfinite(target_features).all():
            raise ValueError("Target embeddings contain NaN or Inf.")

        measures = _require("pymdma.image.measures.synthesis_val", "pymdma")
        ImprovedPrecision, ImprovedRecall = measures.ImprovedPrecision, measures.ImprovedRecall
        Density, Coverage = measures.Density, measures.Coverage
        metrics = {
            "improved_precision": ImprovedPrecision,
            "improved_recall": ImprovedRecall,
            "density": Density,
            "coverage": Coverage,
        }

        results = {}
        context = {}

        for name, metric_class in metrics.items():
            evaluator = metric_class(
                k=k,
                metric=metric,
                n_workers=n_workers,
            )

            result = evaluator.compute(
                reference_features,
                target_features,
                context=context,
            )

            dataset_score, instance_scores = result.value

            results[name] = {
                "dataset": dataset_score,
                "instance": instance_scores,
            }

        return results
