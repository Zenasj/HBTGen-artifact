#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster Markdown Pipeline (LLM-driven)

This module generates Markdown artifacts per (CSV file, cluster label),
by calling an external LLM client function `ask_gpt_langchain(...)`.

Typical workflow:
1) Discover CSV files under a directory and filter them by max label threshold.
2) For each CSV and each label in a label range, extract code files/blocks as a question payload.
3) Call `ask_gpt_langchain(...)` with (question, model, api_key, api_base, prompt_file).
4) Save Markdown to disk, skipping files that already exist (optional).
5) (Extended) For each (CSV, label), also generate a trigger-focused Markdown:
     "{csv_stem}-{label_id}-trigger.md"
   which contains framework/model operation snippets extracted via AST.

Note:
- This module does not directly manage proxy/env settings.
- The caller is responsible for providing correct model/api_key/api_base/prompt_file.
"""

from __future__ import annotations

import ast
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.cluster_and_feature_generate.helper import (
    get_max_cluster_label,
    extract_files_by_label,
    process_files_by_label,
)
from src.helper.llm_langchain_client import ask_gpt_langchain


# ======================================================================
# Data classes
# ======================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class CsvInfo:
    abs_path: str
    filename: str
    max_label: int


@dataclass(frozen=True)
class GenerationStats:
    processed_csv_files: int
    processed_labels: int
    generated_markdowns: int
    skipped_existing: int
    failed: int


# ======================================================================
# File utilities
# ======================================================================

def read_text_file(file_path: str) -> str:
    """
    Read a text file and return content.
    Raises FileNotFoundError if it does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(content: str, file_path: str, overwrite: bool = True) -> None:
    """
    Write content to a file. Create parent directories if needed.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if (not overwrite) and os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_markdown_path(output_dir: str, csv_file: str, label_id: int) -> str:
    """
    Compute the markdown file path for a given CSV file and label id.
    Format: "{csv_basename_without_ext}-{label_id}.md"
    """
    base = os.path.basename(csv_file)
    stem = os.path.splitext(base)[0]
    return os.path.join(output_dir, f"{stem}-{label_id}.md")


def build_trigger_markdown_path(output_dir: str, csv_file: str, label_id: int) -> str:
    """
    Compute the trigger markdown file path for a given CSV file and label id.
    Format: "{csv_basename_without_ext}-{label_id}-trigger.md"
    """
    base = os.path.basename(csv_file)
    stem = os.path.splitext(base)[0]
    return os.path.join(output_dir, f"{stem}-{label_id}-trigger.md")


# ======================================================================
# CSV discovery
# ======================================================================

def list_csv_files(
    directory: str,
    min_max_label_threshold: int = 0,
) -> List[CsvInfo]:
    """
    Walk a directory recursively and return a list of CSV files with their max cluster label.

    Args:
        directory: root directory to search.
        min_max_label_threshold: only keep CSV files whose max label > threshold.

    Returns:
        List[CsvInfo]
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    csv_infos: List[CsvInfo] = []
    for root, _, files in os.walk(directory):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            abs_path = os.path.abspath(os.path.join(root, fn))
            try:
                max_label = int(get_max_cluster_label(abs_path))
            except Exception as e:
                print(f"[WARN] Failed to get max label for CSV: {abs_path}. Reason: {e}")
                continue

            print(f"[INFO] CSV: {fn}, path: {abs_path}, max_label: {max_label}")
            if max_label > min_max_label_threshold:
                csv_infos.append(CsvInfo(abs_path=abs_path, filename=fn, max_label=max_label))

    return csv_infos


# ======================================================================
# Extraction helpers (LLM question payload)
# ======================================================================

def build_question_for_label(
    csv_file: str,
    label_id: int,
    extract: bool = False,
) -> str:
    """
    Extract code/files for a given label from the CSV.

    Thin wrapper over extract_files_by_label(...) to keep the pipeline consistent.
    """
    return extract_files_by_label(csv_file, label_id, extract=extract)


# ======================================================================
# Trigger snippet extraction utilities (AST-based)
# ======================================================================

CONTEXT_WINDOW_LINES: int = 3

API_ROOT_NAMES: Set[str] = {
    "torch",
    "nn",
    "tf",
    "tensorflow",
    "keras",
    "tvm",
    "oneflow",
    "jax",
}


def _clean_code(code: str) -> str:
    """Collapse multiple blank lines into one and strip leading/trailing whitespace."""
    return re.sub(r"\n\s*\n+", "\n", code.strip())


def _is_nn_module_base(base: ast.expr) -> bool:
    """Check whether a base class is nn.Module / torch.nn.Module."""
    # Name: Module（simple fallback）
    if isinstance(base, ast.Name):
        return base.id in {"Module"}

    # Attribute: nn.Module / torch.nn.Module
    if isinstance(base, ast.Attribute):
        root = base
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in {"nn", "torch"}:
            return True

    return False


def _read_python_file_split_nn(filepath: Path) -> Tuple[str, str]:
    """
    Rough read of Python file, splitting into:
      - nn_module_code: class definitions inheriting nn.Module / torch.nn.Module
      - non_nn_module_code: all remaining lines (imports, functions, script body, etc.)

    If parsing fails, treat entire file as non_nn_module_code.
    """
    src = filepath.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "", src

    lines = src.splitlines(keepends=True)
    n = len(lines)
    is_nn_module_line = [False] * n

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if any(_is_nn_module_base(b) for b in node.bases):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                start = max(0, start)
                end = min(n, end)
                for i in range(start, end):
                    is_nn_module_line[i] = True

    nn_module_lines = []
    non_nn_module_lines = []
    for i, line in enumerate(lines):
        if is_nn_module_line[i]:
            nn_module_lines.append(line)
        else:
            non_nn_module_lines.append(line)

    nn_module_code = "".join(nn_module_lines)
    non_nn_module_code = "".join(non_nn_module_lines)
    return nn_module_code, non_nn_module_code


def _get_root_name(node: ast.AST) -> str:
    """
    For an expression (esp. call.func / attribute.value), return the left-most Name.id.

    Examples:
        torch.nn.Linear(...) -> "torch"
        nn.Conv2d(...)       -> "nn"
        model.to("cuda")     -> "model"
    """
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _snippet_has_real_api(snippet: str) -> bool:
    """
    Determine whether a snippet contains a real API call:
    - Not just comments/blank lines.
    - Has at least one Call whose root is in API_ROOT_NAMES.
    """
    # Remove comments and blank lines
    code_lines = [
        l for l in snippet.splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    if not code_lines:
        return False

    cleaned = "\n".join(code_lines)
    try:
        sub_tree = ast.parse(cleaned)
    except SyntaxError:
        # Fallback: check if any API_ROOT_NAMES appears textually
        return any(root in cleaned for root in API_ROOT_NAMES)

    for sub in ast.walk(sub_tree):
        if isinstance(sub, ast.Call):
            root_name = _get_root_name(sub.func)
            if root_name in API_ROOT_NAMES:
                return True
    return False


def extract_api_operation_snippets(code: str) -> Set[str]:
    """
    From non nn.Module code, extract "framework/model operation" snippets, deduplicated.

    Steps:
      1) Use AST to locate all Calls whose root is in API_ROOT_NAMES (torch/tf/tvm/...).
      2) Mark those lines as interesting; expand CONTEXT_WINDOW_LINES around them.
      3) In that context region, strip imports/def/class/main/decorators; keep others.
      4) Split into contiguous blocks, filter out blocks that:
           - are empty, or
           - contain only comments / blanks, or
           - do not contain any real API calls.
    """
    snippets: Set[str] = set()
    if not code.strip():
        return snippets

    lines = code.splitlines()
    n = len(lines)
    if n == 0:
        return snippets

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: simple line-based heuristic
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if any(root in stripped for root in API_ROOT_NAMES):
                snippets.add(stripped)
        return snippets

    # 1. Mark definition lines (import/def/class and main-blocks)
    definition_line = [False] * n

    for node in tree.body:
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            start = getattr(node, "lineno", 1) - 1
            end = getattr(node, "end_lineno", start + 1)
            start = max(0, start)
            end = min(n, end)
            for i in range(start, end):
                definition_line[i] = True

            # Decorators are also considered definition lines
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for dec in getattr(node, "decorator_list", []):
                    dec_lineno = getattr(dec, "lineno", None)
                    if dec_lineno is not None:
                        idx = dec_lineno - 1
                        if 0 <= idx < n:
                            definition_line[idx] = True

        # Detect "if __name__ == '__main__'"-style main blocks
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                start = max(0, start)
                end = min(n, end)
                for i in range(start, end):
                    definition_line[i] = True

    # 2. Mark lines that contain interesting API calls
    interesting_line = [False] * n

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            root_name = _get_root_name(node.func)
            if root_name in API_ROOT_NAMES:
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                start = max(0, start)
                end = min(n, end)
                for i in range(start, end):
                    interesting_line[i] = True

    # 3. Expand context around interesting lines
    context_line = [False] * n
    for i in range(n):
        if interesting_line[i]:
            left = max(0, i - CONTEXT_WINDOW_LINES)
            right = min(n - 1, i + CONTEXT_WINDOW_LINES)
            for j in range(left, right + 1):
                context_line[j] = True

    # 4. Split into contiguous blocks, filter with _snippet_has_real_api
    current_block: List[str] = []

    def flush_block():
        nonlocal current_block
        if not current_block:
            return
        snippet = "\n".join(current_block)
        snippet = _clean_code(snippet)
        if snippet and _snippet_has_real_api(snippet):
            snippets.add(snippet)
        current_block = []

    for idx in range(n):
        if not context_line[idx]:
            flush_block()
            continue

        if definition_line[idx]:
            # Skip definition lines even if in context
            continue

        if not lines[idx].strip():
            flush_block()
            continue

        current_block.append(lines[idx])

    flush_block()
    return snippets


def generate_trigger_markdown_for_label(
    csv_file: str,
    label_id: int,
    output_dir: str,
    *,
    skip_if_exists: bool = True,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    Generate a trigger-focused Markdown file for one (csv_file, label_id).

    Logic:
      - Read CSV, collect all file paths with the given label_id (label >= 0).
      - Read each file and extract non-nn.Module code.
      - From non-nn.Module code, extract framework/model operation snippets.
      - Deduplicate snippets and write them as a Markdown with sections:

          {csv_stem}-{label_id}-trigger.md

    Notes:
      - CSV 中的路径如果是绝对路径，直接使用；
      - 如果是相对路径，则相对于 CSV 所在目录解析。
    """
    trigger_md_path = build_trigger_markdown_path(output_dir, csv_file, label_id)

    if skip_if_exists and os.path.isfile(trigger_md_path):
        print(f"[INFO] Skip existing trigger markdown: {trigger_md_path}")
        return True, trigger_md_path

    method_name = Path(csv_file).stem
    csv_dir = Path(csv_file).parent

    label_paths: List[Path] = []
    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                filepath_str, label_str = row[0].strip(), row[1].strip()
                try:
                    lbl = int(label_str)
                except ValueError:
                    continue
                if lbl != label_id or lbl < 0:
                    continue

                p = Path(filepath_str)
                if not p.is_absolute():
                    # 相对路径时，相对于 CSV 所在目录解析
                    p = csv_dir / p
                label_paths.append(p)
    except Exception as e:
        print(f"[WARN] Failed to read CSV for trigger markdown: {csv_file}. Reason: {e}")
        return False, trigger_md_path

    if not label_paths:
        print(f"[INFO] No file paths found for label={label_id} in CSV={csv_file}")
        return False, trigger_md_path

    all_snippets: Set[str] = set()
    for file_path in label_paths:
        if not file_path.is_file():
            continue
        try:
            _, non_nn_code = _read_python_file_split_nn(file_path)
            snippets = extract_api_operation_snippets(non_nn_code)
            all_snippets.update(snippets)
        except Exception as e:
            print(f"[WARN] Failed to extract snippets from {file_path}. Reason: {e}")
            continue

    if not all_snippets:
        print(f"[INFO] No trigger snippets for {csv_file} label={label_id}; trigger MD not created.")
        return False, trigger_md_path

    os.makedirs(os.path.dirname(trigger_md_path), exist_ok=True)
    try:
        with open(trigger_md_path, "w", encoding="utf-8") as out_f:
            out_f.write(f"# {method_name} – cluster {label_id} (trigger snippets)\n\n")
            out_f.write(f"- Total unique operation snippets: **{len(all_snippets)}**\n\n")

            for idx, snippet in enumerate(sorted(all_snippets), start=1):
                out_f.write(f"## Snippet {idx}\n\n")
                out_f.write("```python\n")
                out_f.write(snippet)
                out_f.write("\n```\n\n")

        print(f"[INFO] Saved trigger markdown: {trigger_md_path}")
        return True, trigger_md_path
    except Exception as e:
        print(f"[ERROR] Failed to write trigger markdown: {trigger_md_path}. Reason: {e}")
        return False, trigger_md_path


# ======================================================================
# Markdown generation pipeline (LLM-based + trigger MD)
# ======================================================================

def generate_markdown_for_label(
    output_dir: str,
    csv_file: str,
    label_id: int,
    *,
    model: str,
    api_key: str,
    api_base: str,
    prompt_file: Optional[str] = None,
    sys_prompt: str = "You are an expert in deep learning and software engineering.",
    extract: bool = False,
    skip_if_exists: bool = True,
    overwrite: bool = False,
    enable_trigger_markdown: bool = True,
    skip_trigger_if_exists: bool = True,
) -> Tuple[bool, str]:
    """
    Generate a markdown file for one (csv_file, label_id) by:

      1) Extracting code/files for the given label.
      2) Passing them as `question` to ask_gpt_langchain(...).
      3) Saving the returned Markdown content.
      4) (Optional) Generating a trigger-focused Markdown side-by-side.

    Args:
        output_dir: Directory where Markdown files are stored.
        csv_file: Path to the cluster CSV file.
        label_id: Target cluster label.
        model: LLM model name (OpenAI-compatible).
        api_key: API key for the LLM endpoint.
        api_base: Base URL for the LLM endpoint.
        prompt_file: Optional prompt/template file path (e.g., task requirements in Markdown).
        sys_prompt: System prompt passed to ask_gpt_langchain.
        extract: Passed to extract_files_by_label (e.g., only model classes).
        skip_if_exists: Skip generating if the markdown file already exists.
        overwrite: If True, overwrite existing file even if skip_if_exists is False.
        enable_trigger_markdown: If True, also generate "{stem}-{label_id}-trigger.md".
        skip_trigger_if_exists: Skip trigger Markdown if already exists.

    Returns:
        (success, markdown_path) – success only refers to the main (LLM) markdown.
    """
    md_path = build_markdown_path(output_dir, csv_file, label_id)

    if skip_if_exists and os.path.isfile(md_path):
        print(f"[INFO] Skip existing markdown: {md_path}")
        # 这里仍然可以选择生成 trigger markdown（如果需要的话）
        if enable_trigger_markdown:
            generate_trigger_markdown_for_label(
                csv_file=csv_file,
                label_id=label_id,
                output_dir=output_dir,
                skip_if_exists=skip_trigger_if_exists,
                overwrite=overwrite,
            )
        return True, md_path

    try:
        question_text = build_question_for_label(csv_file, label_id, extract=extract)
        if not question_text or not question_text.strip():
            print(f"[WARN] Empty question_text for {csv_file} label={label_id}. "
                  f"LLM will be called with empty content.")

        # Call the LLM client; all non-essential parameters use defaults.
        markdown = ask_gpt_langchain(
            question=question_text,
            model=model,
            api_key=api_key,
            api_base=api_base,
            sys_prompt=sys_prompt,
            prompt_file=prompt_file,
        )

        if markdown is None:
            raise ValueError("ask_gpt_langchain returned None")

        write_text_file(markdown, md_path, overwrite=overwrite)
        print(f"[INFO] Saved markdown: {md_path}")

        # After main markdown is generated, optionally generate trigger markdown.
        if enable_trigger_markdown:
            generate_trigger_markdown_for_label(
                csv_file=csv_file,
                label_id=label_id,
                output_dir=output_dir,
                skip_if_exists=skip_trigger_if_exists,
                overwrite=overwrite,
            )

        return True, md_path

    except Exception as e:
        print(f"[ERROR] Failed to generate markdown for {csv_file} label={label_id}. Reason: {e}")
        return False, md_path


def generate_markdowns_for_csv_list(
    output_dir: str,
    csv_list: Sequence[CsvInfo],
    *,
    model: str,
    api_key: str,
    api_base: str,
    prompt_file: Optional[str] = None,
    sys_prompt: str = "You are an expert in deep learning and software engineering.",
    min_label: int = 0,
    max_label: Optional[int] = None,
    extract: bool = False,
    skip_if_exists: bool = True,
    overwrite: bool = False,
    enable_trigger_markdown: bool = True,
    skip_trigger_if_exists: bool = True,
) -> GenerationStats:
    """
    Generate markdown files for all labels of all CSV files in csv_list using ask_gpt_langchain,
    and optionally generate trigger-focused Markdown side-by-side.

    Label range:
      - For each CSV, target_max_label = min(max_label, csv.max_label) if max_label is set,
        otherwise target_max_label = csv.max_label.

    Args:
        output_dir: Directory to store markdown artifacts.
        csv_list: List of CsvInfo.
        model: LLM model name (OpenAI-compatible).
        api_key: API key for the LLM endpoint.
        api_base: Base URL for the LLM endpoint.
        prompt_file: Optional prompt/template file path (e.g., cluster_template_prompt.md).
        sys_prompt: System prompt injected as the first message.
        min_label: Inclusive start label.
        max_label: Inclusive end label (global cap). If None, use each CSV max label.
        extract: Passed to extract_files_by_label.
        skip_if_exists: Skip if markdown already exists.
        overwrite: If True, overwrite existing file even if skip_if_exists is False.
        enable_trigger_markdown: If True, also generate trigger markdowns.
        skip_trigger_if_exists: Skip trigger markdown if already exists.

    Returns:
        GenerationStats
    """
    os.makedirs(output_dir, exist_ok=True)

    processed_csv = 0
    processed_labels = 0
    generated = 0
    skipped = 0
    failed = 0

    for info in csv_list:
        processed_csv += 1

        target_max = info.max_label if max_label is None else min(max_label, info.max_label)
        if target_max < min_label:
            print(f"[INFO] Skip CSV with target_max < min_label: {info.abs_path}")
            continue

        for label_id in range(min_label, target_max + 1):
            processed_labels += 1
            print(f"[INFO] Processing: {info.filename}, label={label_id}/{target_max}")

            md_path = build_markdown_path(output_dir, info.abs_path, label_id)
            if skip_if_exists and os.path.isfile(md_path):
                skipped += 1
                # 这里也可以同步生成 trigger markdown（如果开启）
                if enable_trigger_markdown:
                    generate_trigger_markdown_for_label(
                        csv_file=info.abs_path,
                        label_id=label_id,
                        output_dir=output_dir,
                        skip_if_exists=skip_trigger_if_exists,
                        overwrite=overwrite,
                    )
                continue

            ok, _ = generate_markdown_for_label(
                output_dir=output_dir,
                csv_file=info.abs_path,
                label_id=label_id,
                model=model,
                api_key=api_key,
                api_base=api_base,
                prompt_file=prompt_file,
                sys_prompt=sys_prompt,
                extract=extract,
                skip_if_exists=skip_if_exists,
                overwrite=overwrite,
                enable_trigger_markdown=enable_trigger_markdown,
                skip_trigger_if_exists=skip_trigger_if_exists,
            )
            if ok:
                generated += 1
            else:
                failed += 1

    return GenerationStats(
        processed_csv_files=processed_csv,
        processed_labels=processed_labels,
        generated_markdowns=generated,
        skipped_existing=skipped,
        failed=failed,
    )


# ======================================================================
# Optional: post-processing utilities (kept from original script)
# ======================================================================

def process_files_by_label_to_dir(
    csv_file: str,
    output_dir: str,
    label_id: int,
) -> None:
    """
    Wrapper around process_files_by_label(...) to mirror original CLI behavior.
    Use only if you still need to dump extracted files per label into a directory.

    NOTE: This function does not do any LLM work.
    """
    filename_without_suffix = os.path.splitext(os.path.basename(csv_file))[0]
    out_dir = os.path.join(output_dir, filename_without_suffix)
    os.makedirs(out_dir, exist_ok=True)
    process_files_by_label(csv_file, label_id, out_dir)
