#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster CSV Utilities

This module provides helper functions to:
- Read cluster assignment CSV files and compute max cluster label.
- Extract code contents for files belonging to a given cluster label.
- Split Python source code into (model-related class code, non-model code) using AST parsing.
- Write paired code blocks to Markdown.
- Generate per-label Markdown files from a cluster CSV.

Design goals:
- Pure helper module (importable), no CLI side-effects.
- English-only messages.
- Robust to multiple CSV schemas.
"""

from __future__ import annotations

import ast
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------
# AST extraction helpers (embedded as requested)
# -----------------------------

MODEL_BASE_FULL_NAMES = {
    "nn.Module",
    "torch.nn.Module",
    "tf.keras.Model",
    "keras.Model",
    "layers.Layer",
    "tf.keras.layers.Layer",
    "keras.layers.Layer",
}

MODEL_BASE_SHORT_NAMES = {"Module", "Model"}


def extract_nn_module_classes(code_content: str) -> str:
    """
    Extract all class definitions that inherit from nn.Module / torch.nn.Module / tf.keras.Model / keras.Model
    (and common Layer bases). If none match, fall back to extracting all class definitions.

    Returns:
        A concatenated source string produced by ast.unparse (Python 3.9+).
    """
    tree = ast.parse(code_content)
    class_nodes: List[ast.ClassDef] = []
    fallback_nodes: List[ast.ClassDef] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fallback_nodes.append(node)
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    # Build dotted name from attribute chain
                    full_attr = []
                    cur = base
                    while isinstance(cur, ast.Attribute):
                        full_attr.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        full_attr.append(cur.id)
                    full_attr = full_attr[::-1]
                    full_name = ".".join(full_attr)

                    if full_name in MODEL_BASE_FULL_NAMES:
                        class_nodes.append(node)
                        break

                elif isinstance(base, ast.Name) and base.id in MODEL_BASE_SHORT_NAMES:
                    class_nodes.append(node)
                    break

    selected_nodes = class_nodes if class_nodes else fallback_nodes
    return "\n\n".join(ast.unparse(cls) for cls in selected_nodes)


def extract_non_nn_module_code(code_content: str) -> str:
    """
    Extract code that is NOT part of model-related class definitions, plus all non-class top-level code.

    If no model class exists, keep all class definitions.

    Returns:
        A concatenated source string produced by ast.unparse (Python 3.9+).
    """
    tree = ast.parse(code_content)
    keep_nodes: List[ast.AST] = []
    model_class_nodes: List[ast.ClassDef] = []
    all_class_nodes: List[ast.ClassDef] = []

    def is_model_base(base: ast.AST) -> bool:
        if isinstance(base, ast.Attribute):
            # Note: this is a lightweight check; it matches common "nn.Module" patterns.
            full_name = f"{getattr(base.value, 'id', '')}.{base.attr}"
            return full_name in MODEL_BASE_FULL_NAMES
        if isinstance(base, ast.Name):
            return base.id in {"Module", "Model", "Layer"}
        return False

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            all_class_nodes.append(node)
            if any(is_model_base(b) for b in node.bases):
                model_class_nodes.append(node)
            else:
                keep_nodes.append(node)
        else:
            keep_nodes.append(node)

    # Fallback: if no model class found, keep all classes
    if len(model_class_nodes) == 0:
        for cls in all_class_nodes:
            if cls not in keep_nodes:
                keep_nodes.append(cls)

    return "\n\n".join(ast.unparse(n) for n in keep_nodes)


# -----------------------------
# CSV schema support
# -----------------------------

@dataclass(frozen=True)
class ClusterCsvSchema:
    path_col: str
    label_col: str


KNOWN_SCHEMAS: Sequence[ClusterCsvSchema] = (
    ClusterCsvSchema(path_col="Vector Name", label_col="Cluster Label"),
    ClusterCsvSchema(path_col="file_path", label_col="cluster_label"),
)


def detect_cluster_csv_schema(df: pd.DataFrame) -> ClusterCsvSchema:
    """
    Detect which known CSV schema is used.
    Raises ValueError if none match.
    """
    for sch in KNOWN_SCHEMAS:
        if sch.path_col in df.columns and sch.label_col in df.columns:
            return sch
    raise ValueError(
        f"Unsupported CSV schema. Expected one of: "
        f"{[(s.path_col, s.label_col) for s in KNOWN_SCHEMAS]}, got columns={list(df.columns)}"
    )


# -----------------------------
# Core helpers
# -----------------------------

def get_max_cluster_label(csv_file_path: str) -> int:
    """
    Get the maximum cluster label in a cluster CSV file.

    Supports multiple schemas (see KNOWN_SCHEMAS).

    Returns:
        int: max label
    """
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)
    schema = detect_cluster_csv_schema(df)

    max_label = df[schema.label_col].max()
    if pd.isna(max_label):
        raise ValueError(f"Label column is empty: {schema.label_col} in {csv_file_path}")
    return int(max_label)


def extract_files_by_label(
    csv_file_path: str,
    cluster_label: int,
    extract: bool = False,
    strip_prefix: Optional[str] = None,
) -> str:
    """
    Extract file contents for all files whose label matches `cluster_label`, and return as a single string.

    Args:
        csv_file_path: path to cluster CSV.
        cluster_label: target label.
        extract: if True, extract only model-related classes via AST; otherwise include full file content.
        strip_prefix: optional prefix to strip from printed file paths for readability.

    Returns:
        A single string containing concatenated file blocks.
    """
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    df = pd.read_csv(csv_file_path)
    schema = detect_cluster_csv_schema(df)

    filtered = df[df[schema.label_col] == cluster_label][schema.path_col].tolist()
    if not filtered:
        return ""

    out_parts: List[str] = []
    for file_path in filtered:
        if not isinstance(file_path, str) or not file_path.strip():
            continue

        if not os.path.isfile(file_path):
            print(f"[WARN] File not found: {file_path}")
            continue

        display_path = file_path
        if strip_prefix and display_path.startswith(strip_prefix):
            display_path = display_path.replace(strip_prefix, "", 1)

        out_parts.append(f"\n================== FILE: {display_path} ==================\n")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if extract:
            content = extract_nn_module_classes(content)
        out_parts.append(content)

    return "".join(out_parts)


def read_python_file(filepath: str) -> Tuple[str, str]:
    """
    Read a Python file and return:
      - model_class_code: extracted model-related class definitions (fallback to all classes)
      - non_model_code: code excluding model classes (fallback to keep all if no model class)

    Returns empty strings on read/parse failure.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code_str = f.read()
        nn_module_code = extract_nn_module_classes(code_str)
        non_nn_module_code = extract_non_nn_module_code(code_str)
        return nn_module_code, non_nn_module_code
    except Exception as e:
        print(f"[WARN] Failed to read/parse file: {filepath}. Reason: {e}")
        return "", ""


def write_code_lists_to_markdown(
    list_a: Sequence[str],
    list_b: Sequence[str],
    output_path: str,
    prefix_a: str = "Section A",
    prefix_b: str = "Section B",
    code_fence_lang: str = "python",
) -> None:
    """
    Write two aligned lists of code blocks into a Markdown file, organized as two sections.

    Args:
        list_a: first group of code blocks.
        list_b: second group of code blocks.
        output_path: markdown output path.
        prefix_a: header prefix for list A items.
        prefix_b: header prefix for list B items.
        code_fence_lang: language tag for fenced code blocks.
    """
    if len(list_a) != len(list_b):
        raise ValueError(f"List lengths must match: {len(list_a)} vs {len(list_b)}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def clean_code(code: str) -> str:
        # Collapse multiple blank lines into a single blank line
        return re.sub(r"\n\s*\n+", "\n", (code or "").strip())

    with open(output_path, "w", encoding="utf-8") as f:
        for i, code in enumerate(list_a):
            f.write(f"{prefix_a} {i}:\n")
            f.write(f"```{code_fence_lang}\n")
            f.write(clean_code(code) + "\n")
            f.write("```\n\n")

        f.write("\n---\n\n")

        for j, code in enumerate(list_b):
            f.write(f"{prefix_b} {j}:\n")
            f.write(f"```{code_fence_lang}\n")
            f.write(clean_code(code) + "\n")
            f.write("```\n\n")

    print(f"[INFO] Markdown written: {output_path}")


def process_files_by_label(
    csv_path: str,
    target_label: int,
    md_output_dir: str,
    md_filename: Optional[str] = None,
    prefix_a: str = "Model-related classes",
    prefix_b: str = "Non-model code (potential trigger context)",
) -> str:
    """
    Read all files with the given label from a cluster CSV and write a Markdown file containing:
      - extracted model class code (list A)
      - extracted non-model code (list B)

    Args:
        csv_path: cluster CSV path.
        target_label: label to extract.
        md_output_dir: output directory for markdown.
        md_filename: output markdown filename. If None, uses "{target_label}.md".
        prefix_a: title prefix for section A.
        prefix_b: title prefix for section B.

    Returns:
        The path to the generated markdown file.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    schema = detect_cluster_csv_schema(df)

    # Extract rows for label
    matched_rows = df[df[schema.label_col] == target_label]
    file_paths = matched_rows[schema.path_col].tolist()

    nn_list: List[str] = []
    non_nn_list: List[str] = []
    matched_count = 0

    for fp in file_paths:
        if not isinstance(fp, str) or not fp.strip():
            continue
        if not os.path.isfile(fp):
            print(f"[WARN] File not found: {fp}")
            continue

        nn_code, non_nn_code = read_python_file(fp)
        nn_list.append(nn_code)
        non_nn_list.append(non_nn_code)
        matched_count += 1

    os.makedirs(md_output_dir, exist_ok=True)
    md_name = md_filename if md_filename is not None else f"{target_label}.md"
    md_path = os.path.join(md_output_dir, md_name)

    write_code_lists_to_markdown(
        nn_list,
        non_nn_list,
        md_path,
        prefix_a=prefix_a,
        prefix_b=prefix_b,
        code_fence_lang="python",
    )

    print(f"[INFO] Processed files: {matched_count} for label={target_label}")
    return md_path
