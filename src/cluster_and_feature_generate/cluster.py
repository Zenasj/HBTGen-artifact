#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering and labeling utilities for code embedding vectors.

Main use cases:
1) Load vector dictionaries saved by the encoder (torch-saved dicts).
2) Perform dimensionality reduction (PCA/UMAP).
3) Run clustering (DBSCAN/HDBSCAN/KMeans optional).
4) Save cluster assignments to CSV.
5) (Optional) Visualize clusters in 2D/3D scatter plots.
6) Classify issues by text summaries and copy corresponding .py files into category folders.

This module is designed to be imported and called by other scripts (no __main__ required).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Dict, Any, Union

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

# Optional deps: umap-learn and hdbscan.
# If they are missing, we raise a clear error when those methods are invoked.
try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

# Matplotlib is optional for headless environments: only used when visualize=True.
import matplotlib.pyplot as plt


# -----------------------------
# Data classes
# -----------------------------

@dataclass(frozen=True)
class ClusteringArtifacts:
    """Paths of artifacts produced by a clustering run."""
    output_csv: str
    output_plot: Optional[str] = None


@dataclass(frozen=True)
class ClusteringResult:
    """In-memory output of a clustering run."""
    labels: np.ndarray
    reduced_embeddings: np.ndarray
    artifacts: Optional[ClusteringArtifacts] = None


# -----------------------------
# Vector IO (decoupled from encoder)
# -----------------------------

def load_vector_dict(vector_path: str) -> Dict[str, np.ndarray]:
    """
    Load a vector dictionary from a torch-saved .pth file (recommended), or a text fallback.

    The encoder refactor earlier saved torch dicts using torch.save({path: np.ndarray}, file).
    This loader focuses on that format.

    Returns:
        Dict[str, np.ndarray]
    """
    import torch  # local import to avoid hard dependency for non-IO usage

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    loaded = torch.load(vector_path, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(f"Loaded object is not a dict: {type(loaded)}")

    valid: Dict[str, np.ndarray] = {}
    for k, v in loaded.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(f"Value for key {k} is not np.ndarray, got {type(v)}")
        if np.isnan(v).any():
            continue
        if np.all(v == 0):
            continue
        valid[k] = v
    return valid


def merge_vector_dicts(vector_dicts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Merge multiple dicts; later keys overwrite earlier ones."""
    out: Dict[str, np.ndarray] = {}
    for d in vector_dicts:
        out.update(d)
    return out


def load_vectors_from_directories(
    directories: Sequence[str],
    vector_file_name: str,
) -> Dict[str, np.ndarray]:
    """
    Load and merge vector dictionaries from a set of directories.

    Args:
        directories: list of directories.
        vector_file_name: e.g., "microsoft-codebert-base-vectors.pth" or your custom name.

    Returns:
        merged Dict[path -> vector]
    """
    dicts: List[Dict[str, np.ndarray]] = []
    for d in directories:
        vector_path = os.path.join(d, vector_file_name)
        if os.path.isfile(vector_path):
            dicts.append(load_vector_dict(vector_path))
    return merge_vector_dicts(dicts)


# -----------------------------
# Dimensionality reduction
# -----------------------------

def reduce_dimensionality(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings into n_components dimensions.

    Supported methods:
        - "pca"
        - "umap"

    Returns:
        reduced embeddings as np.ndarray with shape (N, n_components)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N, D], got shape={embeddings.shape}")

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn is not installed. Install with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Invalid reduction method. Use 'pca' or 'umap'.")

    print(f"[INFO] Reduced embeddings shape: {reduced.shape}")
    return reduced


# -----------------------------
# Clustering methods
# -----------------------------

def cluster_dbscan(
    reduced_embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    DBSCAN clustering. Noise points are labeled as -1.
    """
    if reduced_embeddings.ndim != 2:
        raise ValueError(f"reduced_embeddings must be 2D, got {reduced_embeddings.shape}")

    algo = DBSCAN(eps=eps, min_samples=min_samples)
    labels = algo.fit_predict(reduced_embeddings)

    unique = np.unique(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] DBSCAN labels: {unique}")
    print(f"[INFO] DBSCAN discovered clusters: {n_clusters} (noise label: -1)")
    return labels


def cluster_hdbscan(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> np.ndarray:
    """
    HDBSCAN clustering. Noise points are labeled as -1.
    """
    if hdbscan is None:
        raise ImportError("hdbscan is not installed. Install with: pip install hdbscan")
    if reduced_embeddings.ndim != 2:
        raise ValueError(f"reduced_embeddings must be 2D, got {reduced_embeddings.shape}")

    algo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = algo.fit_predict(reduced_embeddings)

    unique = np.unique(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] HDBSCAN labels: {unique}")
    print(f"[INFO] HDBSCAN discovered clusters: {n_clusters} (noise label: -1)")
    return labels


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    KMeans clustering on original embeddings (or reduced embeddings if you prefer).
    """
    algo = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = algo.fit_predict(embeddings)
    return labels


# -----------------------------
# Evaluation metrics (optional)
# -----------------------------

def evaluate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Internal metric: Silhouette score in [-1, 1]. Higher is better.
    Note: Silhouette is not well-defined for 1 cluster or when all points are noise.
    """
    if len(set(labels)) <= 1:
        raise ValueError("Silhouette score requires at least 2 clusters.")
    score = silhouette_score(embeddings, labels)
    print(f"[INFO] Silhouette score: {score:.4f}")
    return score


def evaluate_adjusted_rand(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    External metric: Adjusted Rand Index (requires ground-truth labels).
    """
    score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"[INFO] Adjusted Rand Index: {score:.4f}")
    return score


def evaluate_hcv(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Homogeneity, Completeness, V-measure (requires ground-truth labels).
    """
    h = homogeneity_score(true_labels, predicted_labels)
    c = completeness_score(true_labels, predicted_labels)
    v = v_measure_score(true_labels, predicted_labels)
    print(f"[INFO] Homogeneity={h:.4f}, Completeness={c:.4f}, V-measure={v:.4f}")
    return h, c, v


# -----------------------------
# Visualization (optional)
# -----------------------------

def plot_clusters(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot 2D or 3D clusters using matplotlib.

    Args:
        reduced_embeddings: shape (N, 2) or (N, 3).
        labels: shape (N,).
        title: plot title.
        output_path: if provided, save plot to this path.
        show: if True, display interactively (may block on servers).

    Returns:
        output_path if saved else None
    """
    if reduced_embeddings.shape[1] not in (2, 3):
        raise ValueError("Plotting requires reduced_embeddings with 2 or 3 components.")

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # Basic coloring without relying on label->color being contiguous
    cmap = plt.get_cmap("tab20")

    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        pts = reduced_embeddings[mask]
        color = "black" if lab == -1 else cmap(idx / max(1, len(unique_labels) - 1))
        if reduced_embeddings.shape[1] == 2:
            plt.scatter(pts[:, 0], pts[:, 1], s=18, label=f"Cluster {lab}", c=[color])
        else:
            # Minimal 3D support: fallback to 2D projection if needed
            plt.scatter(pts[:, 0], pts[:, 1], s=18, label=f"Cluster {lab}", c=[color])

    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    saved = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        saved = output_path
        print(f"[INFO] Saved plot to: {output_path}")

    if show:
        plt.show()
    plt.close()
    return saved


def visualize_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = False,
    random_state: int = 42,
) -> Optional[str]:
    """
    t-SNE visualization to 2D. Useful for qualitative inspection.
    """
    reducer = TSNE(n_components=2, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return plot_clusters(
        reduced_embeddings=reduced,
        labels=labels,
        title="Cluster Visualization (t-SNE)",
        output_path=output_path,
        show=show,
    )


# -----------------------------
# Cluster result persistence
# -----------------------------

def save_cluster_results_csv(
    vector_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    output_csv: str,
) -> str:
    """
    Save (file_path, cluster_label) pairs to CSV.

    Args:
        vector_dict: key=file path, value=vector
        labels: clustering labels aligned with vector_dict iteration order
        output_csv: output path

    Returns:
        output_csv
    """
    names = list(vector_dict.keys())
    if len(names) != len(labels):
        raise ValueError(f"Length mismatch: {len(names)} vectors vs {len(labels)} labels")

    df = pd.DataFrame({"file_path": names, "cluster_label": labels.astype(int)})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved cluster CSV to: {output_csv}")
    return output_csv


# -----------------------------
# Higher-level pipelines
# -----------------------------

def filter_outliers_isolation_forest(
    embeddings: np.ndarray,
    file_paths: List[str],
    contamination: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove outliers using IsolationForest.

    Returns:
        (filtered_embeddings, filtered_file_paths)
    """
    if embeddings.shape[0] != len(file_paths):
        raise ValueError("embeddings and file_paths must have the same length.")

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    flags = clf.fit_predict(embeddings)  # -1 for outliers
    keep_idx = [i for i, f in enumerate(flags) if f != -1]

    filtered_embeddings = embeddings[keep_idx]
    filtered_paths = [file_paths[i] for i in keep_idx]
    print(f"[INFO] Outlier filtering: kept={len(keep_idx)}, removed={len(file_paths) - len(keep_idx)}")
    return filtered_embeddings, filtered_paths


def find_optimal_kmeans_k(
    vector_dict: Dict[str, np.ndarray],
    max_k: int = 10,
    use_pca: bool = True,
    pca_components: int = 50,
    filter_outliers: bool = True,
    contamination: float = 0.1,
    show_plot: bool = False,
    random_state: int = 42,
) -> List[Tuple[int, float]]:
    """
    Sweep k in [2..max_k] and compute silhouette scores to find a reasonable K for KMeans.

    Returns:
        list of (k, silhouette_score)
    """
    embeddings = np.array(list(vector_dict.values()))
    file_paths = list(vector_dict.keys())

    if filter_outliers:
        embeddings, file_paths = filter_outliers_isolation_forest(
            embeddings, file_paths, contamination=contamination, random_state=random_state
        )

    if use_pca:
        pca = PCA(n_components=min(pca_components, embeddings.shape[1]), random_state=random_state)
        embeddings = pca.fit_transform(embeddings)
        print(f"[INFO] PCA reduced shape for k-sweep: {embeddings.shape}")

    results: List[Tuple[int, float]] = []
    scores: List[float] = []

    for k in range(2, max_k + 1):
        labels = cluster_kmeans(embeddings, n_clusters=k, random_state=random_state)
        score = silhouette_score(embeddings, labels)
        results.append((k, score))
        scores.append(score)
        print(f"[INFO] KMeans k={k}, silhouette={score:.4f}")

    if show_plot:
        plt.plot(range(2, max_k + 1), scores)
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette score")
        plt.title("KMeans: silhouette vs k")
        plt.tight_layout()
        plt.show()
        plt.close()

    return results


def run_density_clustering_pipeline(
    vector_dict: Dict[str, np.ndarray],
    method: str = "dbscan",
    reduction: str = "umap",
    n_components: int = 2,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 3,
    hdbscan_min_cluster_size: int = 5,
    output_dir: Optional[str] = None,
    output_prefix: str = "model",
    save_plot: bool = False,
    show_plot: bool = False,
    random_state: int = 42,
) -> ClusteringResult:
    """
    End-to-end pipeline:
    1) Extract embeddings and reduce dimensionality (PCA/UMAP).
    2) Cluster using DBSCAN or HDBSCAN.
    3) Save cluster results CSV.
    4) Optionally save a plot.

    Args:
        vector_dict: {file_path: vector}
        method: "dbscan" | "hdbscan"
        reduction: "pca" | "umap"
        n_components: 2 or 3 recommended for plotting
        output_dir: where to save CSV/plot; if None, no artifacts are saved.
        output_prefix: prefix used in file naming.
        save_plot: if True, save plot image (requires output_dir)
        show_plot: if True, display plot interactively
    """
    if not vector_dict:
        raise ValueError("vector_dict is empty; nothing to cluster.")

    file_paths = list(vector_dict.keys())
    embeddings = np.array(list(vector_dict.values()))

    reduced = reduce_dimensionality(
        embeddings=embeddings,
        method=reduction,
        n_components=n_components,
        random_state=random_state,
    )

    if method == "dbscan":
        labels = cluster_dbscan(reduced, eps=dbscan_eps, min_samples=dbscan_min_samples)
        algo_name = "dbscan"
    elif method == "hdbscan":
        labels = cluster_hdbscan(reduced, min_cluster_size=hdbscan_min_cluster_size)
        algo_name = "hdbscan"
    else:
        raise ValueError("Unsupported method. Use 'dbscan' or 'hdbscan'.")

    artifacts: Optional[ClusteringArtifacts] = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{output_prefix}-{algo_name}-cluster_results.csv")
        save_cluster_results_csv(vector_dict, labels, csv_path)

        plot_path = None
        if save_plot:
            plot_path = os.path.join(output_dir, f"{output_prefix}-{algo_name}-clusters.png")
            plot_clusters(
                reduced_embeddings=reduced,
                labels=labels,
                title=f"{algo_name.upper()} clustering ({output_prefix})",
                output_path=plot_path,
                show=show_plot,
            )
        else:
            if show_plot:
                plot_clusters(
                    reduced_embeddings=reduced,
                    labels=labels,
                    title=f"{algo_name.upper()} clustering ({output_prefix})",
                    output_path=None,
                    show=True,
                )

        artifacts = ClusteringArtifacts(output_csv=csv_path, output_plot=plot_path)

    return ClusteringResult(labels=labels, reduced_embeddings=reduced, artifacts=artifacts)


# -----------------------------
# Text-based classification & copying
# -----------------------------

def classify_and_copy_python_files_by_text(
    txt_dir: str,
    py_dir: str,
    keyword_map: Dict[str, List[str]],
    dst_base_dir: str,
    txt_suffix: str = ".txt",
    py_suffix: str = ".py",
    default_category: str = "unknown",
    lowercase_match: bool = True,
) -> Dict[str, int]:
    """
    Classify txt files by keywords in their content, and copy corresponding .py files.

    It assumes each txt file name (without suffix) matches a .py file in py_dir.

    Args:
        txt_dir: directory containing text summaries.
        py_dir: directory containing python files to copy.
        keyword_map: category -> list of keywords
        dst_base_dir: output root directory; subfolders per category will be created.
        default_category: folder for no-match cases
        lowercase_match: if True, match using lowercased content + lowercased keywords

    Returns:
        stats dict: category -> count
    """
    if not os.path.isdir(txt_dir):
        raise NotADirectoryError(f"txt_dir is not a directory: {txt_dir}")
    if not os.path.isdir(py_dir):
        raise NotADirectoryError(f"py_dir is not a directory: {py_dir}")

    os.makedirs(dst_base_dir, exist_ok=True)
    stats: Dict[str, int] = {k: 0 for k in keyword_map.keys()}
    stats[default_category] = 0

    # Normalize keyword map
    norm_map: Dict[str, List[str]] = {}
    for cat, kws in keyword_map.items():
        if lowercase_match:
            norm_map[cat] = [kw.lower() for kw in kws]
        else:
            norm_map[cat] = kws

    for fn in os.listdir(txt_dir):
        if not fn.endswith(txt_suffix):
            continue

        txt_path = os.path.join(txt_dir, fn)
        issue_name = os.path.splitext(os.path.basename(txt_path))[0]
        py_path = os.path.join(py_dir, issue_name + py_suffix)

        if not os.path.isfile(py_path):
            # Skip missing python file but keep visibility
            print(f"[WARN] Missing python file for {txt_path}: {py_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        if lowercase_match:
            content = content.lower()

        matched_category: Optional[str] = None
        for cat, kws in norm_map.items():
            if any(kw in content for kw in kws):
                matched_category = cat
                break

        if matched_category is None:
            matched_category = default_category

        target_dir = os.path.join(dst_base_dir, matched_category)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(py_path, os.path.join(target_dir, os.path.basename(py_path)))
        stats[matched_category] = stats.get(matched_category, 0) + 1

    print(f"[INFO] Classification stats: {stats}")
    return stats


def run_clustering_for_vector_directories(
    directory_list: Iterable[Union[str, Path]],
    vector_file_pattern: str = "*-vectors.pth",
    method: str = "dbscan",
    reduction: str = "umap",
    n_components: int = 3,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 3,
    save_plot: bool = True,
    show_plot: bool = False,
) -> Dict[Path, Any]:
    """
    For each directory in `directory_list`, find all vector .pth files that match
    `vector_file_pattern`, run the density clustering pipeline, and return the
    results in a dictionary keyed by the .pth file path.

    - directory_list: list of directories that potentially contain vector .pth files.
    - vector_file_pattern: glob pattern for vector files, default '*-vectors.pth'.
    - output_dir: for each run, is set to the directory where the .pth file is located.
    - output_prefix: for each run, is derived from the pth file name by stripping
      the '-vectors.pth' suffix if present, otherwise using the stem.
    """
    results: Dict[Path, Any] = {}

    for d in directory_list:
        dir_path = Path(d).expanduser().resolve()
        if not dir_path.is_dir():
            print(f"[WARN] Skipping non-existing directory: {dir_path}")
            continue

        print(f"[INFO] Scanning directory for vector files: {dir_path}")

        # Find all pth files that match the pattern, e.g., "*-vectors.pth"
        pth_files = sorted(dir_path.glob(vector_file_pattern))
        if not pth_files:
            print(f"[INFO] No vector files found in: {dir_path}")
            continue

        for pth_file in pth_files:
            output_dir = pth_file.parent

            # Derive output_prefix from file name
            stem = pth_file.stem  # e.g., 'microsoft-codebert-base-vectors'
            if stem.endswith("-vectors"):
                output_prefix = stem[: -len("-vectors")]
            else:
                output_prefix = stem

            print(
                f"[INFO] Processing vector file: {pth_file.name} "
                f"(output_dir={output_dir}, output_prefix={output_prefix})"
            )

            # Here we reuse your existing helper. It will load only the vectors
            # from the current directory and the current file name.
            vectors = load_vectors_from_directories(
                directories=[dir_path],
                vector_file_name=pth_file.name,
            )

            res = run_density_clustering_pipeline(
                vector_dict=vectors,
                method=method,
                reduction=reduction,
                n_components=n_components,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                output_dir=str(output_dir),
                output_prefix=output_prefix,
                save_plot=save_plot,
                show_plot=show_plot,
            )

            # Store the result keyed by the concrete pth path
            results[pth_file] = res

            # Optional: log artifacts if you like
            try:
                artifacts = getattr(res, "artifacts", None)
                if artifacts is not None:
                    print(f"[INFO] Artifacts generated for {pth_file.name}: {artifacts}")
            except Exception as e:
                print(f"[WARN] Failed to access artifacts for {pth_file.name}: {e}")

    return results
