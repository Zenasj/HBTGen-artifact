#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Vectorizer Utilities

This module encodes Python code (optionally only model-related classes) into embedding vectors
using HuggingFace Transformers models (e.g., CodeBERT, CodeT5, InCoder, GraphCodeBERT, CodeLlama, Qwen).

Key characteristics:
- Callable functions intended to be imported by other scripts.
"""

from __future__ import annotations

import os
import ast
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

# Disable parallel tokenizer warnings/noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Configuration
# -----------------------------

SUPPORTED_MODELS = {
    "microsoft/codebert-base",
    "Salesforce/codet5-base",
    "facebook/incoder-1B",
    "microsoft/graphcodebert-base",
    "codellama/CodeLlama-7b-hf",
    "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass(frozen=True)
class VectorizationResult:
    """Summary for a single model run."""
    model_name: str
    processed_directories: List[str]
    saved_files: List[str]


# -----------------------------
# AST utilities
# -----------------------------

MODEL_BASE_NAMES = {
    # PyTorch
    "nn.Module",
    "torch.nn.Module",
    # TensorFlow / Keras
    "tf.keras.Model",
    "keras.Model",
    # Layers
    "layers.Layer",
    "tf.keras.layers.Layer",
    "keras.layers.Layer",
}

MODEL_BASE_SHORT_NAMES = {"Module", "Model", "Layer"}


def _attr_to_full_name(attr: ast.AST) -> Optional[str]:
    """
    Convert an ast.Attribute chain to a dot-separated string name.
    Example: torch.nn.Module -> "torch.nn.Module"
    """
    if not isinstance(attr, ast.Attribute):
        return None
    parts = []
    node = attr
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


def extract_model_related_classes(code_content: str) -> str:
    """
    Extract classes that inherit from nn.Module / torch.nn.Module / tf.keras.Model / keras.Model / Layer, etc.
    If no matching model classes are found, fall back to extracting ALL class definitions.

    Returns:
        Source code string composed by joining extracted class blocks via ast.unparse (Python 3.9+).
    """
    tree = ast.parse(code_content)
    matched: List[ast.ClassDef] = []
    all_classes: List[ast.ClassDef] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            all_classes.append(node)
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    full_name = _attr_to_full_name(base)
                    if full_name and full_name in MODEL_BASE_NAMES:
                        matched.append(node)
                        break
                elif isinstance(base, ast.Name) and base.id in MODEL_BASE_SHORT_NAMES:
                    matched.append(node)
                    break

    selected = matched if matched else all_classes
    return "\n\n".join(ast.unparse(cls) for cls in selected)


def extract_non_model_code(code_content: str) -> str:
    """
    Extract code that does NOT belong to model-related class definitions, plus all non-class top-level code.
    If no model classes exist, keep all classes.

    Returns:
        Concatenated code blocks via ast.unparse.
    """
    tree = ast.parse(code_content)

    keep_nodes: List[ast.AST] = []
    model_classes: List[ast.ClassDef] = []
    all_classes: List[ast.ClassDef] = []

    def is_model_base(base: ast.AST) -> bool:
        if isinstance(base, ast.Attribute):
            full_name = _attr_to_full_name(base)
            return bool(full_name and full_name in MODEL_BASE_NAMES)
        if isinstance(base, ast.Name):
            return base.id in MODEL_BASE_SHORT_NAMES
        return False

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            all_classes.append(node)
            if any(is_model_base(b) for b in node.bases):
                model_classes.append(node)
            else:
                keep_nodes.append(node)
        else:
            keep_nodes.append(node)

    if len(model_classes) == 0:
        # No model class found: keep all classes
        for cls in all_classes:
            if cls not in keep_nodes:
                keep_nodes.append(cls)

    return "\n\n".join(ast.unparse(n) for n in keep_nodes)


# -----------------------------
# IO utilities
# -----------------------------

def save_numpy_dict(numpy_dict: Dict[str, np.ndarray], file_path: str) -> None:
    """
    Save a {path: np.ndarray} dict to disk using torch.save.

    Note:
        For portability, vectors are stored as numpy arrays.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(numpy_dict, file_path)
    print(f"[INFO] Saved vector dictionary to: {file_path}")


def load_numpy_dict(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load a saved vector dictionary.

    Supports:
    - torch.save binary file (recommended)
    - text format: "path<TAB>vec_numbers"

    Invalid vectors are removed (all zeros or contains NaN).
    """
    print(f"[INFO] Loading vectors from: {file_path}")

    valid: Dict[str, np.ndarray] = {}
    removed_zero, removed_nan = 0, 0

    try:
        loaded = torch.load(file_path, weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError("Loaded object is not a dict.")
        for k, v in loaded.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(f"Value for key {k} is not np.ndarray, got {type(v)}")
            if np.all(v == 0):
                removed_zero += 1
                continue
            if np.isnan(v).any():
                removed_nan += 1
                continue
            valid[k] = v
        print(f"[INFO] Loaded via torch.load. Removed zero={removed_zero}, nan={removed_nan}. Kept={len(valid)}")
        return valid
    except Exception as e:
        print(f"[WARN] torch.load failed; trying text format. Reason: {e}")

    # Text fallback
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                path, vec_str = parts
                vec = np.array([float(x) for x in vec_str.split()], dtype=np.float32)
                if np.all(vec == 0):
                    removed_zero += 1
                    continue
                if np.isnan(vec).any():
                    removed_nan += 1
                    continue
                valid[path] = vec
        print(f"[INFO] Loaded via text. Removed zero={removed_zero}, nan={removed_nan}. Kept={len(valid)}")
        return valid
    except Exception as e2:
        print(f"[ERROR] Failed to read vector file: {file_path}. Reason: {e2}")
        return {}


# -----------------------------
# Model loading / encoding
# -----------------------------

def _infer_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_name: str,
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, torch.nn.Module, torch.device]:
    """
    Load tokenizer and model for supported HuggingFace checkpoints.

    No DataParallel wrapping is used here by default; caller can wrap if needed.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {sorted(SUPPORTED_MODELS)}")

    trust = ("Qwen/" in model_name)

    if model_name == "Salesforce/codet5-base":
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    elif model_name in {"microsoft/codebert-base", "microsoft/graphcodebert-base"}:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        mdl = AutoModel.from_pretrained(model_name)

    else:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust)

    dev = _infer_device(device)
    mdl = mdl.to(dev).eval()
    return tok, mdl, dev

@torch.no_grad()
def encode_code_snippet(
    code_snippet: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Encode a code snippet into a single embedding vector (mean pooling).

    Returns:
        torch.Tensor of shape [hidden_size]
    """
    if not code_snippet.strip():
        # Empty snippet -> return zeros (caller can decide to drop it)
        hidden_size = getattr(model.config, "hidden_size", 768)
        return torch.zeros(hidden_size, device=device)

    if model_name == "Salesforce/codet5-base":
        inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        real_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        encoder_out = real_model.encoder(**inputs)
        emb = encoder_out.last_hidden_state.mean(dim=1).squeeze(0)
        return emb

    # Most encoder/decoder or causal LMs can provide last_hidden_state or hidden_states
    if model_name in {"microsoft/codebert-base", "microsoft/graphcodebert-base"}:
        inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0)
        return emb

    # Causal LM branch: request hidden states
    inputs = tokenizer(
        code_snippet,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
    ).to(device)

    out = model(**inputs, output_hidden_states=True)
    last = out.hidden_states[-1]
    emb = last.mean(dim=1).squeeze(0)
    return emb


def encode_code_files_to_vectors(
    directory: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    model: torch.nn.Module,
    device: torch.device,
    extract_mode: str = "model_classes",
) -> Dict[str, np.ndarray]:
    """
    Encode all .py files under directory into vectors.

    Args:
        extract_mode:
            - "model_classes": only extract model-related class definitions (fallback to all classes).
            - "non_model_code": keep code excluding model classes (fallback to keep all if no model class).
            - "full_file": encode entire file content.

    Returns:
        dict: {file_path: vector(np.ndarray)}
    """
    vectors: Dict[str, np.ndarray] = {}

    if not os.path.exists(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return vectors

    for root, _, files in os.walk(directory):
        for fn in files:
            if not fn.endswith(".py"):
                continue

            full_path = os.path.join(root, fn)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()

                if extract_mode == "model_classes":
                    snippet = extract_model_related_classes(code)
                elif extract_mode == "non_model_code":
                    snippet = extract_non_model_code(code)
                elif extract_mode == "full_file":
                    snippet = code
                else:
                    raise ValueError(f"Unknown extract_mode: {extract_mode}")

                emb = encode_code_snippet(
                    code_snippet=snippet,
                    model_name=model_name,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                )

                vectors[full_path] = emb.detach().to(torch.float32).cpu().numpy()

                print(f"[INFO] Encoded: {full_path}")

            except Exception as e:
                print(f"[WARN] Failed to encode {full_path}: {e}")

    return vectors


# -----------------------------
# Directory traversal
# -----------------------------

def iter_target_directories(root_dir: str) -> List[str]:
    """
    If root_dir contains .py files, return [root_dir] only.
    Otherwise return [root_dir] + all subdirectories (recursive).
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"root_dir is not a directory: {root_dir}")

    has_py = any(
        f.endswith(".py") and os.path.isfile(os.path.join(root_dir, f))
        for f in os.listdir(root_dir)
    )
    if has_py:
        return [root_dir]

    dirs = [root_dir]
    for cur_root, subdirs, _ in os.walk(root_dir):
        for d in subdirs:
            dirs.append(os.path.join(cur_root, d))
    return dirs


# -----------------------------
# Main callable API
# -----------------------------

def build_and_save_code_vectors(
    root_dir: str,
    model_names: List[str],
    save: bool = True,
    vector_file_suffix: str = "-vectors.pth",
    extract_mode: str = "model_classes",
    device: Optional[str] = None,
) -> List[VectorizationResult]:
    """
    Main entrypoint intended to be imported and called by other scripts.

    It iterates directories under root_dir (root only if it already contains .py files),
    encodes .py files into vectors for each model, and saves vectors into
    sibling result directories instead of the original input directories.

    Args:
        root_dir: Root directory to process.
        model_names: A list of HF model names (must be in SUPPORTED_MODELS).
        save: Whether to save vector dicts to disk.
        vector_file_suffix: Saved filename format, e.g. "-vectors.pth". Final name is "{safe_model_name}{suffix}".
        extract_mode: "model_classes" | "non_model_code" | "full_file".
        device: Optional, e.g. "cuda", "cpu", "cuda:0". If None, auto.

    Returns:
        A list of VectorizationResult, one per model.
    """
    target_dirs = iter_target_directories(root_dir)
    print(f"[INFO] Found {len(target_dirs)} directories to process.")

    results: List[VectorizationResult] = []

    for model_name in model_names:
        print(f"\n[INFO] ===== Current model: {model_name} =====")
        tokenizer, model, dev = load_model_and_tokenizer(model_name=model_name, device=device)

        saved_files: List[str] = []
        safe_model_name = model_name.replace("/", "-")
        out_name = f"{safe_model_name}{vector_file_suffix}"

        for d in target_dirs:
            print(f"[INFO] Processing directory: {d}")

            vectors = encode_code_files_to_vectors(
                directory=d,
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                device=dev,
                extract_mode=extract_mode,
            )

            if save:
                # Build sibling result directory
                d_path = Path(d).resolve()
                result_dir = d_path.parent / f"{d_path.name}-abstract-result"
                result_dir.mkdir(exist_ok=True)

                out_path = result_dir / out_name
                print(f"[INFO] Saving vectors to: {out_path}")

                save_numpy_dict(vectors, str(out_path))
                saved_files.append(str(out_path))

        results.append(
            VectorizationResult(
                model_name=model_name,
                processed_directories=target_dirs,
                saved_files=saved_files,
            )
        )

    return results
