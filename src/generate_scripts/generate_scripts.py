#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generation & Evaluation Pipeline (with bug classification)

This module provides:

1) LLM-based generation utilities:
   - Generate model code from clustered Markdown features.
   - Generate trigger functions for known bugs based on generated models and
     clustered issue descriptions.
   - Enhance bug-triggering scripts by running them and refining them based on
     the observed outputs.

2) Bug classification pipeline:
   - From each bugs_trigger directory, select the best-version script per group.
   - Execute each script (trigger_known_bugs), capture outputs.
   - Ask an LLM to classify the behavior as [normal] / [abnormal].
   - Move scripts and markdown reports into normal / abnormal folders.

3) A top-level orchestration loop:
   - Repeatedly runs model generation, trigger generation, enhancement, and
     classification under a wall-clock time budget.

Design principles:
- All LLM calls go through `ask_gpt_langchain(...)`.
- No hard-coded API keys or proxy settings.
- All prompt files are passed into this module as parameters and forwarded to
  the LLM client; this module does not read prompt contents itself.
- No absolute paths or machine-specific configuration.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import io
import keyword
import hashlib
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.helper.llm_langchain_client import ask_gpt_langchain
from src.cluster_and_feature_generate.helper import read_python_file
from src.generate_scripts.helper import (
    extract_python_code,
    save_string_to_python_file,
    find_max_version,
    merge_markdown_files,
)

# =============================================================================
# Low-level utilities for dynamic execution & file naming
# =============================================================================


def _sanitize_mod_name(raw_name: str, path: Path, prefix: str = "m_") -> str:
    """
    Sanitize a file stem into a valid Python module name and append a short hash
    derived from the absolute file path to avoid collisions.
    """
    name = re.sub(r"[^0-9A-Za-z_]", "_", raw_name)
    name = re.sub(r"_+", "_", name).strip("_")

    if (not name) or (not re.match(r"[A-Za-z_]", name[0])) or keyword.iskeyword(name):
        name = f"{prefix}{name}" if name else f"{prefix}mod"

    while (not name.isidentifier()) or keyword.iskeyword(name):
        name = f"{prefix}{name}"

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{name}_{digest}"


def run_single_trigger_known_bugs(
    file_path: str,
    return_dict: Dict[str, str],
    function_name: str = "trigger_known_bugs",
) -> None:
    """
    Dynamically import a Python file, execute a function (default: trigger_known_bugs),
    and capture its stdout/stderr into return_dict[file_path].
    """
    key = file_path
    filename = os.path.basename(file_path)
    raw_module_name = filename[:-3]

    safe_module_name = _sanitize_mod_name(raw_module_name, Path(file_path))

    spec = importlib.util.spec_from_file_location(safe_module_name, file_path)
    if spec is None or spec.loader is None:
        return_dict[key] = "[ERROR] Failed to create import spec."
        return

    module = importlib.util.module_from_spec(spec)

    try:
        sys.modules[safe_module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, function_name):
            return_dict[key] = f"[ERROR] Module does not define '{function_name}'."
            return

        trigger_func = getattr(module, function_name)

        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stdout_capture):
            trigger_func()

        return_dict[key] = stdout_capture.getvalue()

    except Exception as exc:  # noqa: BLE001
        return_dict[key] = (
            f"[ERROR] Exception while executing {function_name}: {exc}\n"
            f"{traceback.format_exc()}"
        )


def save_code_to_timestamped_file(content: str, name_part: str, output_dir: str) -> str:
    """
    Save content to a Python file with filename: {name_part}_{timestamp}.py

    Returns:
        Full path to the saved file.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{name_part}_{timestamp}.py"
    full_path = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"[INFO] Saved file: {full_path}")
    return full_path


def save_versioned_python_file(
    code_str: str,
    filename: str,
    dir_path: str,
    version: int,
) -> str:
    """
    Save Python code to a file with a version suffix inside curly braces in the name.

    Examples:
        filename="model.py", version=3        -> model_{3}.py
        filename="model_{1}.py", version=2    -> model_{2}.py

    Trailing underscores before the version braces are cleaned up.
    """
    base_name, ext = os.path.splitext(filename)

    version_pattern = re.compile(r"(.*?)(_+\{\d+\})$")
    match = version_pattern.match(base_name)

    if match:
        prefix = match.group(1).rstrip("_")
        new_base = f"{prefix}_{{{version}}}"
    else:
        cleaned_base = base_name.rstrip("_")
        new_base = f"{cleaned_base}_{{{version}}}"

    new_filename = new_base + ext
    full_path = os.path.join(dir_path, new_filename)

    os.makedirs(dir_path, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as file:
        file.write(code_str)

    print(f"[INFO] Saved versioned file: {full_path}")
    return full_path


def parse_python_file_path(file_path: str, root_dir: str) -> Tuple[str, List[int], str, int]:
    """
    Given a full .py file path and a root directory, extract:
        - model_folder: the parent directory under root_dir
        - label_list: list of integers encoded after timestamp in filename
        - filename_wo_ext: filename without extension (and version suffix)
        - version: integer from trailing '{n}' in filename if exists, else 0

    Expected filename pattern (before .py):
        YYYYMMDD_HHMMSS[_label1[_label2...]][_{version}]
    """
    if not file_path or not root_dir:
        raise ValueError(f"Invalid path arguments: file_path={file_path}, root_dir={root_dir}")

    rel_parts = os.path.relpath(file_path, root_dir).split(os.sep)
    if len(rel_parts) < 2:
        raise ValueError(f"Unexpected path structure: {file_path}")

    model_folder = rel_parts[-2]
    filename = os.path.basename(file_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    version = 0
    version_pattern = re.compile(r"\{(\d+)\}$")
    version_match = version_pattern.search(filename_wo_ext)
    if version_match:
        version = int(version_match.group(1))
        filename_wo_ext = filename_wo_ext[: version_match.start()]

    label_pattern = re.compile(r"^\d{8}_\d{6}((?:_\d+)*)$")
    match = label_pattern.match(filename_wo_ext)
    if match:
        tail = match.group(1)
        if tail:
            label_list = [int(x) for x in tail.strip("_").split("_")]
        else:
            label_list = []
    else:
        label_list = []

    return model_folder, label_list, filename_wo_ext, version


# =============================================================================
# Feature-based model generation helpers
# =============================================================================


def _generate_md_from_labels(
    species: str,
    quantity: int,
    versions: Dict[str, int],
    cluster_md_dir: str,
) -> Tuple[str, List[int]]:
    """
    Randomly sample `quantity` label indices for a given species and merge
    the corresponding Markdown files into a single string.

    Assumes Markdown files are named "{species}-{i}.md" in `cluster_md_dir`.
    If a companion trigger file "{species}-{i}-trigger.md" exists, its content
    is appended after the main cluster description.
    """
    if species not in versions:
        raise KeyError(f"Species key not found in versions: {species}")

    max_size = versions[species]
    if quantity > max_size + 1:
        raise ValueError(f"Requested quantity {quantity} exceeds max available {max_size + 1}.")

    sampled_indices = random.sample(range(max_size + 1), quantity)
    md_file_paths: List[str] = []

    for idx in sampled_indices:
        base = f"{species}-{idx}.md"
        md_path = os.path.join(cluster_md_dir, base)
        md_file_paths.append(md_path)

        trigger_path = os.path.join(cluster_md_dir, f"{species}-{idx}-trigger.md")
        if os.path.exists(trigger_path):
            md_file_paths.append(trigger_path)

    merged_md = merge_markdown_files(md_file_paths)
    return merged_md, sampled_indices


def _generate_code_block_from_labels(
    label_list: List[int],
    csv_path: str,
    *,
    trigger_md_dir: Optional[str] = None,
    species: Optional[str] = None,
) -> str:
    """
    For each label in label_list, collect the non-nn.Module code parts from the
    corresponding Python files (according to the given CSV mapping) and assemble
    a multi-block code string in Markdown format.

    If `trigger_md_dir` and `species` are provided and a file
    "{species}-{label}-trigger.md" exists under `trigger_md_dir`, its content
    is appended to the block as additional context.
    """

    def clean_code(code: str) -> str:
        return re.sub(r"\n\s*\n+", "\n", code.strip())

    non_nn_module_code_list: List[List[str]] = []

    for label_index in label_list:
        block_snippets: List[str] = []

        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header

            for row in reader:
                if len(row) < 2:
                    continue

                filepath, label_str = row[0].strip(), row[1].strip()
                try:
                    label = int(label_str)
                except ValueError:
                    continue

                if label == label_index:
                    nn_module_code, non_nn_module_code = read_python_file(filepath)
                    block_snippets.append(non_nn_module_code)

        # Optionally append trigger markdown content
        if trigger_md_dir is not None and species is not None:
            trigger_path = os.path.join(
                trigger_md_dir,
                f"{species}-{label_index}-trigger.md",
            )
            if os.path.exists(trigger_path):
                try:
                    with open(trigger_path, "r", encoding="utf-8") as trigger_file:
                        trigger_text = trigger_file.read()
                    block_snippets.append("\n\n# Existing trigger description\n")
                    block_snippets.append(trigger_text)
                except OSError:
                    print(f"[WARN] Failed to read trigger markdown: {trigger_path}")

        non_nn_module_code_list.append(block_snippets)

    code_block = ""
    for block_index, blocks in enumerate(non_nn_module_code_list):
        code_block += f"This is block {block_index}:\n"
        code_block += "```python\n"
        code_block += "".join(blocks)
        code_block += "```\n\n"

    return clean_code(code_block)


def generate_from_labels(
    versions: Dict[str, int],
    loop_times: int,
    max_quantity: int,
    result_path: str,
    prompt_file: Optional[str],
    issue_path: str,
    cluster_md_dir: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Legacy variant: for each species and each loop, sample labels, merge Markdown
    features, append code blocks from issues, and ask the LLM to generate models.

    Note:
        - `prompt_file` is forwarded to `ask_gpt_langchain` and is not read here.
    """
    for _ in range(loop_times):
        for species in versions:
            try:
                csv_path = os.path.join(issue_path, f"{os.path.basename(species)}.csv")
                current_quantity = random.randint(1, max_quantity)

                md_str, label_list = _generate_md_from_labels(
                    species,
                    current_quantity,
                    versions,
                    cluster_md_dir,
                )
                code_block = _generate_code_block_from_labels(
                    label_list,
                    csv_path,
                    trigger_md_dir=cluster_md_dir,
                    species=os.path.basename(species),
                )
                questions = md_str + "\n" + code_block

                answer = ask_gpt_langchain(
                    question=questions,
                    model=model,
                    api_key=api_key,
                    api_base=api_base,
                    prompt_file=prompt_file,
                    temperature=temperature,
                )

                if answer:
                    format_answer = extract_python_code(answer)
                    save_string_to_python_file(
                        result_path,
                        os.path.basename(species),
                        format_answer,
                        label_list,
                    )

            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] generate_from_labels error for {species}: {exc}")
                continue


def generate_from_labels_models(
    versions: Dict[str, int],
    loop_times: int,
    max_quantity: int,
    result_path: str,
    prompt_file: Optional[str],
    cluster_md_dir: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Main entrypoint: for each species and loop, sample labels and merge the
    corresponding Markdown feature files, then call the LLM to generate model code.

    Note:
        - `prompt_file` is forwarded to `ask_gpt_langchain` and is not read here.
    """
    for _ in range(loop_times):
        for species in versions:
            try:
                current_quantity = random.randint(1, max_quantity)
                md_str, label_list = _generate_md_from_labels(
                    species,
                    current_quantity,
                    versions,
                    cluster_md_dir,
                )
                questions = md_str

                answer = ask_gpt_langchain(
                    question=questions,
                    model=model,
                    api_key=api_key,
                    api_base=api_base,
                    prompt_file=prompt_file,
                    temperature=temperature,
                )

                if answer:
                    format_answer = extract_python_code(answer)
                    save_string_to_python_file(
                        result_path,
                        os.path.basename(species),
                        format_answer,
                        label_list,
                    )

            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] generate_from_labels_models error for {species}: {exc}")
                continue


# =============================================================================
# Trigger-function generation from models + issues
# =============================================================================


def _generate_from_labels_trigger_functions(
    root_dir: str,
    file_path: str,
    prompt_file: Optional[str],
    result_path: str,
    cluster_md_dir: str,
    issue_csv_root: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Given a single generated model file, infer its labels and the corresponding
    issue snippets (from CSV files under `issue_csv_root`), optionally enriched
    with trigger markdowns under `cluster_md_dir`, then ask the LLM to generate
    trigger functions for known bugs.
    """
    model_folder, labels_list, file_name, _ = parse_python_file_path(
        file_path,
        root_dir,
    )

    csv_path = os.path.join(issue_csv_root, f"{model_folder}.csv")
    code_block = _generate_code_block_from_labels(
        labels_list,
        csv_path,
        trigger_md_dir=cluster_md_dir,
        species=model_folder,
    )

    with open(file_path, "r", encoding="utf-8") as file:
        model_str = file.read()

    model_str = "This is model file:\n" + model_str
    questions = model_str + "\n" + code_block

    try:
        answer = ask_gpt_langchain(
            question=questions,
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_file=prompt_file,
            temperature=temperature,
        )

        if answer:
            format_answer = extract_python_code(answer)
            save_code_to_timestamped_file(
                format_answer,
                file_name,
                os.path.join(result_path, model_folder),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] _generate_from_labels_trigger_functions error for {file_path}: {exc}")


def generate_from_labels_trigger_functions(
    root_dir: str,
    prompt_file: Optional[str],
    result_path: str,
    cluster_md_dir: str,
    issue_csv_root: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Walk through all .py files in root_dir (generated models) and, for each,
    generate trigger functions for known bugs using clustered issue descriptions.

    Parameters:
        root_dir:        Directory that contains generated model .py files.
        prompt_file:     Prompt file forwarded to the LLM client.
        result_path:     Root directory where trigger scripts will be saved.
        cluster_md_dir:  Directory that contains cluster and trigger markdowns.
        issue_csv_root:  Directory that contains per-species CSV files mapping
                         original issues to cluster labels.
    """
    py_paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".py"):
                full_path = os.path.abspath(os.path.join(root, file))
                py_paths.append(full_path)

    for file_path in py_paths:
        _generate_from_labels_trigger_functions(
            root_dir,
            file_path,
            prompt_file,
            result_path,
            cluster_md_dir,
            issue_csv_root,
            api_key=api_key,
            model=model,
            api_base=api_base,
            temperature=temperature,
        )


# =============================================================================
# Enhancement of bug-trigger scripts
# =============================================================================


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _enhance_single_file(
    root_dir: str,
    file_path: str,
    prompt_file: Optional[str],
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
    enable_gpu_selection: bool = True,
) -> Optional[str]:
    """
    Run a single bug-trigger file, feed its code + output to the LLM, and
    save an enhanced version with incremented version index.
    """
    if enable_gpu_selection:
        selected_gpu = select_most_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        print(f"[INFO] (enhance) Selected GPU {selected_gpu} for {file_path}")

    print(f"[INFO] (enhance) Running trigger_known_bugs for {file_path}")

    code_str = _read_text_file(file_path)
    result_dict: Dict[str, str] = {}
    run_single_trigger_known_bugs(file_path, result_dict, "trigger_known_bugs")
    output_str = result_dict.get(file_path, "")

    questions = (
        "------This is file------\n"
        + code_str
        + "\n------This is output------\n"
        + output_str
    )

    try:
        answer = ask_gpt_langchain(
            question=questions,
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_file=prompt_file,
            temperature=temperature,
        )

        if not answer:
            return None

        format_answer = extract_python_code(answer)

        try:
            model_folder, _, file_name, version_index = parse_python_file_path(
                file_path,
                root_dir,
            )
            res_path = save_versioned_python_file(
                format_answer,
                f"{file_name}.py",
                os.path.join(root_dir, model_folder),
                version_index + 1,
            )
        except Exception:  # noqa: BLE001
            stem = Path(file_path).stem
            match = re.match(r"(.+?)(?:_\{(\d+)\})?$", stem)
            file_name_base = match.group(1) if match else stem
            version_index = int(match.group(2) or 0) if match else 0
            res_path = save_versioned_python_file(
                format_answer,
                f"{file_name_base}.py",
                root_dir,
                version_index + 1,
            )
        return res_path

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] _enhance_single_file error for {file_path}: {exc}")
        return None


def enhance_single_file(
    root_dir: str,
    prompt_file: Optional[str],
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
    max_iterations_per_file: int = 5,
) -> None:
    """
    Iteratively enhance all bug-trigger files in root_dir using the LLM.

    For each file:
    - Skip if it already contains a version suffix `{...}`.
    - Up to `max_iterations_per_file`, run `_enhance_single_file` and
      compare ASTs to stop when the code stabilizes.
    """
    py_paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".py"):
                full_path = os.path.abspath(os.path.join(root, file))
                py_paths.append(full_path)

    for file_path in py_paths:
        res_path: Optional[str] = file_path
        print(f"[INFO] Start enhancing: {file_path}")

        if "{" in file_path:
            print(f"[INFO] Already versioned, skip: {file_path}")
            continue

        for _ in range(max_iterations_per_file):
            try:
                with open(res_path, "r", encoding="utf-8") as file:
                    tree1 = ast.parse(file.read())
            except Exception as exc:  # noqa: BLE001
                tree1 = None
                print(f"[WARN] AST parse error (before enhancement) for {res_path}: {exc}")

            if res_path is not None:
                res_path = _enhance_single_file(
                    root_dir,
                    res_path,
                    prompt_file,
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                    temperature=temperature,
                )
            else:
                break

            if res_path is None:
                break

            try:
                with open(res_path, "r", encoding="utf-8") as file:
                    tree2 = ast.parse(file.read())
            except Exception as exc:  # noqa: BLE001
                tree2 = None
                print(f"[WARN] AST parse error (after enhancement) for {res_path}: {exc}")

            if tree1 and tree2 and ast.dump(tree1) == ast.dump(tree2):
                print(f"[INFO] File stabilized, stop iterations: {file_path}")
                break


# =============================================================================
# Single-issue generation variants
# =============================================================================


def generate_from_single_issue(
    prompt_file: Optional[str],
    issue_path: str,
    result_path: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Randomly select a single issue .py file and ask the LLM to generate
    a new model based on it.

    Note:
        - `prompt_file` is forwarded to `ask_gpt_langchain` and is not read here.
    """
    py_files: List[str] = []
    for root, _, files in os.walk(issue_path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    if not py_files:
        print(f"[WARN] No Python files found under issue_path: {issue_path}")
        return

    target_issue = random.choice(py_files)
    merged_str = merge_markdown_files([target_issue])
    questions = merged_str

    answer = ask_gpt_langchain(
        question=questions,
        model=model,
        api_key=api_key,
        api_base=api_base,
        prompt_file=prompt_file,
        temperature=temperature,
    )

    if answer:
        format_answer = extract_python_code(answer)
        file_name = os.path.splitext(os.path.basename(target_issue))[0]
        save_string_to_python_file(result_path, "", format_answer, [file_name])


def _generate_from_issue_trigger_functions(
    file_path: str,
    prompt_file: Optional[str],
    result_path: str,
    issue_path: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    For a single generated model file that is associated with a specific issue
    (deduced from filename), generate trigger functions using the full issue file.
    """
    filename = Path(file_path).name
    issue_part = "_".join(filename.split("_")[2:])
    issue_file = os.path.join(issue_path, issue_part)

    with open(file_path, "r", encoding="utf-8") as file:
        model_str = file.read()
    with open(issue_file, "r", encoding="utf-8") as file:
        code_block = file.read()

    model_str = "This is model file:\n" + model_str
    questions = model_str + "\n" + code_block

    try:
        answer = ask_gpt_langchain(
            question=questions,
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_file=prompt_file,
            temperature=temperature,
        )

        if answer:
            format_answer = extract_python_code(answer)
            save_code_to_timestamped_file(
                format_answer,
                Path(file_path).stem,
                result_path,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] _generate_from_issue_trigger_functions error for {file_path}: {exc}")


def generate_from_issue_trigger_functions(
    root_dir: str,
    prompt_file: Optional[str],
    result_path: str,
    issue_path: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    temperature: float = 0.6,
) -> None:
    """
    Walk through all .py files in root_dir (generated models),
    infer the corresponding issue file from the filename, and ask the LLM
    to generate trigger functions based on model + issue.
    """
    py_paths: List[str] = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".py"):
                full_path = os.path.abspath(os.path.join(root, file))
                py_paths.append(full_path)

    for file_path in py_paths:
        _generate_from_issue_trigger_functions(
            file_path,
            prompt_file,
            result_path,
            issue_path,
            api_key=api_key,
            model=model,
            api_base=api_base,
            temperature=temperature,
        )


# =============================================================================
# Bug classification: GPU selection + best-version selection + LLM labeling
# =============================================================================


def select_most_free_gpu() -> int:
    """
    Use `nvidia-smi` to select the GPU with the most free memory.

    Heuristic:
      - If all GPUs have plenty of free memory and are within a small range,
        randomly choose among the top candidates;
      - Otherwise choose the one with the maximum free memory.

    Returns:
        GPU index (0-based). Falls back to 0 on failure.
    """
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader",
            shell=True,
        )
        free_mem_list = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        if not free_mem_list:
            raise RuntimeError("No GPU info parsed from nvidia-smi output.")

        max_free = max(free_mem_list)
        min_free = min(free_mem_list)

        if max_free > 4000 and (max_free - min_free) <= 100:
            candidates = [i for i, mem in enumerate(free_mem_list) if max_free - mem <= 100]
            best_gpu = random.choice(candidates)
        else:
            best_gpu = max(range(len(free_mem_list)), key=lambda i: free_mem_list[i])

        print(f"[INFO] Selected GPU {best_gpu} (free {free_mem_list[best_gpu]} MB).")
        return best_gpu

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to query GPUs: {exc}. Falling back to GPU 0.")
        return 0


def _handle_bug_trigger_output(
    file_path: str,
    prompt_file: Optional[str],
    api_key: str,
    *,
    model: str,
    api_base: str,
    temperature: float = 0.6,
    enable_gpu_selection: bool = True,
    classify_root_normal: Optional[str] = None,
    classify_root_abnormal: Optional[str] = None,
) -> None:
    """
    For a single bug-trigger file:
      1) Optionally pick the most free GPU and set CUDA_VISIBLE_DEVICES.
      2) Run trigger_known_bugs and capture output.
      3) Ask LLM to classify result ([normal] / [abnormal]).
      4) Copy .py into corresponding subdir and write a .md report.

    When `classify_root_normal` / `classify_root_abnormal` are provided, results are
    stored under:

        classify_root_normal/<species>/<file>.py / .md
        classify_root_abnormal/<species>/<file>.py / .md

    where <species> is the parent directory name of `file_path` under the triggers root.
    Otherwise, fall back to the old layout:

        <file_dir>/normal/<file>.py / .md
        <file_dir>/abnormal/<file>.py / .md
    """
    print(f"======== Running file: {file_path} at {datetime.now().strftime('%H:%M:%S')} ========")

    if enable_gpu_selection:
        selected_gpu = select_most_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)

    # species 取的是触发脚本父目录名，例如 Qwen-Qwen2.5-7B-Instruct-dbscan-cluster_results
    species_name = os.path.basename(file_dir)

    # 目标目录：如果有全局 normal/abnormal 根目录，则优先用它们
    if classify_root_normal is not None and classify_root_abnormal is not None:
        normal_dir = os.path.join(classify_root_normal, species_name)
        abnormal_dir = os.path.join(classify_root_abnormal, species_name)
    else:
        normal_dir = os.path.join(file_dir, "normal")
        abnormal_dir = os.path.join(file_dir, "abnormal")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)

    # 是否已经处理过：只要 global normal/abnormal 里已经有这个文件就跳过
    handled_already = False
    for sub_dir in (normal_dir, abnormal_dir):
        target_path = os.path.join(sub_dir, file_name)
        if os.path.exists(target_path):
            handled_already = True
            break

    if handled_already:
        print(f"[INFO] File {file_path} already handled in global normal/abnormal folders. Skip.")
        return

    print(f"[INFO] File {file_path} will be classified by LLM.")

    code_str = _read_text_file(file_path)

    res_dict: Dict[str, str] = {}
    run_single_trigger_known_bugs(file_path, res_dict, "trigger_known_bugs")
    output_str = res_dict.get(file_path, "")

    questions = (
        "------This is file------\n"
        + code_str
        + "\n------This is output------\n"
        + output_str
    )

    try:
        answer = ask_gpt_langchain(
            question=questions,
            model=model,
            api_key=api_key,
            api_base=api_base,
            prompt_file=prompt_file,
            temperature=temperature,
        )

        if not answer:
            print(f"[WARN] Empty answer for {file_path}.")
            return

        lower_answer = answer.lower()
        category: Optional[str] = None
        if "[abnormal]" in lower_answer:
            category = "abnormal"
        elif "[normal]" in lower_answer:
            category = "normal"
        else:
            print(f"[WARN] Could not find [normal]/[abnormal] markers for {file_path}.")
            return

        if category == "normal":
            dest_dir = normal_dir
        else:
            dest_dir = abnormal_dir

        os.makedirs(dest_dir, exist_ok=True)
        target_path = os.path.join(dest_dir, file_name)
        shutil.copy2(file_path, target_path)
        print(f"[INFO] Copied {file_path} -> {target_path}")

        base, _ = os.path.splitext(target_path)
        md_path = base + ".md"
        with open(md_path, "w", encoding="utf-8") as file:
            file.write(answer + "\n------This is output------\n" + output_str)
        print(f"[INFO] Wrote classification markdown: {md_path}")

    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] _handle_bug_trigger_output error for {file_path}: {exc}")


def handle_bug_trigger_output(
    root_dir: str,
    prompt_file: Optional[str],
    api_key: str,
    *,
    model: str,
    api_base: str,
    temperature: float = 0.6,
    enable_gpu_selection: bool = True,
    classify_root_normal: Optional[str] = None,
    classify_root_abnormal: Optional[str] = None,
) -> List[str]:
    """
    Walk a directory tree, find all .py files (excluding those already under
    normal/abnormal subdirs), and classify them via `_handle_bug_trigger_output`.

    When `classify_root_normal` / `classify_root_abnormal` are provided, results are
    stored under those global roots (with per-species subdirectories).
    """
    py_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        norm_path = os.path.normpath(dirpath)
        parts = norm_path.split(os.sep)
        if "normal" in parts or "abnormal" in parts:
            continue

        for fname in filenames:
            if fname.endswith(".py"):
                full_path = os.path.abspath(os.path.join(dirpath, fname))
                if "normal" in full_path or "abnormal" in full_path:
                    continue
                py_files.append(full_path)

    py_files.sort(key=str.lower)

    for py_file in py_files:
        _handle_bug_trigger_output(
            py_file,
            prompt_file,
            api_key,
            model=model,
            api_base=api_base,
            temperature=temperature,
            enable_gpu_selection=enable_gpu_selection,
            classify_root_normal=classify_root_normal,
            classify_root_abnormal=classify_root_abnormal,
        )

    return py_files


def move_abnormal_files(
    root_dir: str,
    file_ext: str = ".md",
    keyword: str = "[abnormal]",
) -> None:
    """
    Secondary consistency pass:

    - Scan all markdown files under root_dir;
    - If the content contains `keyword` and the file is under a 'normal' directory,
      move both the .md and corresponding .py to the mirrored 'abnormal' directory.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        norm_path = os.path.normpath(dirpath)
        parts = norm_path.split(os.sep)

        for filename in filenames:
            if not filename.endswith(file_ext):
                continue

            md_path = os.path.join(dirpath, filename)
            try:
                with open(md_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                continue

            if keyword not in content:
                continue

            if "normal" not in parts:
                print(f"[INFO] Skip: {md_path} is not under a 'normal' directory.")
                continue

            py_filename = filename.replace(file_ext, ".py")
            py_path = os.path.join(dirpath, py_filename)

            target_dir = dirpath.replace(os.sep + "normal", os.sep + "abnormal")
            os.makedirs(target_dir, exist_ok=True)

            new_md_path = os.path.join(target_dir, filename)
            shutil.move(md_path, new_md_path)
            print(f"[INFO] Moved {md_path} -> {new_md_path}")

            if os.path.exists(py_path):
                new_py_path = os.path.join(target_dir, py_filename)
                shutil.move(py_path, new_py_path)
                print(f"[INFO] Moved {py_path} -> {new_py_path}")
            else:
                print(f"[WARN] No corresponding .py file for {md_path}")


# =============================================================================
# Best-version selection utilities (copy_max_version_py_files)
# =============================================================================

_RE_TAIL = re.compile(
    r"(?:_\{\d+\}|_(?:\d{8}_\d{6}|\d{8}_\d{2}_\d{2}_\d{2}))$"
)


def _split_base_and_version(stem: str) -> Tuple[str, Optional[int]]:
    """
    Strip trailing timestamp segments and version suffixes from a filename stem.

    Return:
        (base, version_or_None)

    Examples:
      'A_20250816_222503_{7}' -> ('A', 7)
      '20250628_223119_44_59_20250628_231658_{5}' -> ('20250628_223119_44_59', 5)
      '20250628_223119_44_59' -> ('20250628_223119_44_59', None)
    """
    version: Optional[int] = None
    s = stem
    while True:
        match = _RE_TAIL.search(s)
        if not match:
            break
        tail = match.group(0)
        if tail.startswith("_{"):
            version = int(tail[2:-1])
        s = s[: match.start()]
    return s, version


def copy_max_version_py_files(
    src_root: str,
    dst_root: str,
    unify_name: bool = False,
) -> None:
    """
    Group files and pick the best version per group, then copy them into dst_root.

    Grouping key: (category, last_dir, base)
      - category: "inconsistency" if path contains "inconsistency" (case-insensitive),
                  otherwise "crash";
      - last_dir: last directory name of the source file;
      - base: filename stem with trailing timestamps and {version} stripped.

    Selection rule:
      - If any versioned files exist in a group, pick the one with highest version.
      - Otherwise, pick a plain candidate (prefer exact `base.py`).

    Copy rule:
      - Each group yields exactly one chosen file.
      - If unify_name = True, the destination filename is base.py;
        otherwise keep the original source filename.
      - If destination file exists, skip to avoid overwriting.
    """
    max_versioned: Dict[Tuple[str, str, str], Tuple[int, str]] = {}
    plain_candidates: Dict[Tuple[str, str, str], str] = {}

    for root, _, files in os.walk(src_root):
        for fname in files:
            if not fname.lower().endswith(".py"):
                continue
            full_path = os.path.join(root, fname)

            stem = fname[:-3]
            base, ver = _split_base_and_version(stem)

            last_dir = os.path.basename(root)
            category = "inconsistency" if "inconsistency" in full_path.lower() else "crash"
            key = (category, last_dir, base)

            if ver is not None:
                if key not in max_versioned or ver > max_versioned[key][0]:
                    max_versioned[key] = (ver, full_path)
            else:
                if key not in plain_candidates or fname == f"{base}.py":
                    plain_candidates[key] = full_path

    chosen: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
    all_keys = set(max_versioned) | set(plain_candidates)
    for key in all_keys:
        base = key[2]
        if key in max_versioned:
            chosen[key] = (base, max_versioned[key][1])
        else:
            chosen[key] = (base, plain_candidates[key])

    for (category, last_dir, base), (_, src_path) in chosen.items():
        dst_dir = os.path.join(dst_root, category, last_dir)
        os.makedirs(dst_dir, exist_ok=True)

        if unify_name:
            dst_file = os.path.join(dst_dir, f"{base}.py")
        else:
            dst_file = os.path.join(dst_dir, os.path.basename(src_path))

        if os.path.exists(dst_file):
            print(f"[INFO] Skip (already exists): {dst_file}")
            continue

        shutil.copy2(src_path, dst_file)
        print(f"[INFO] Copied best-version file: {src_path} -> {dst_file}")


# =============================================================================
# Evaluation time utilities + top-level orchestration loop
# =============================================================================

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


def _format_timedelta(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """
    Format elapsed time since start_time as "HHh MMm SSs".
    """
    if end_time is None:
        end_time = datetime.now()
    elapsed = end_time - start_time
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def log_elapsed_time(start_time: datetime, prefix: str = "Elapsed time") -> None:
    """
    Print the elapsed time since start_time with a given prefix.
    """
    print(f"{prefix}: {_format_timedelta(start_time)}")


def run_evaluation_loop(
    *,
    max_hours: float = 12.0,
    api_key: str,
    # LLM configuration (shared for all stages; you can split if needed)
    llm_model: str,
    llm_api_base: str,
    llm_temperature: float = 0.6,
    # Clustered markdown directory (single root)
    cluster_md_directory: str,
    # Base output directory
    output_root: str,
    # Prompt files
    model_prompt_file: Optional[str] = None,
    trigger_bug_prompt_file: Optional[str] = None,
    enhance_prompt_file: Optional[str] = None,
    classify_prompt_file: Optional[str] = None,
    # Label range (shared for all categories)
    label_min: int = 1,
    label_max: int = 3,
    # Enhancement timeout (not used anymore in the incremental version, kept for API compatibility)
    enhance_timeout_seconds: int = 60 * 30 * 2,
    # Number of passes over all species / encoders
    num_rounds: int = 1,
    # Classification options
    enable_step5_classification: bool = True,
    enable_gpu_selection_for_step5: bool = True,
    unify_name_for_best_version: bool = False,  # kept for compatibility, not used in incremental mode
    # Root directory that contains CSV files produced from clustering
    issue_csv_root: Optional[str] = None,
    # Maximum enhancement iterations per trigger file
    max_iterations_per_file: int = 5,
) -> None:
    """
    Top-level incremental pipeline.

    For up to `num_rounds` and within `max_hours`, this function performs:

        For each round:
          For each species (encoding scheme / cluster prefix):
            1) Sample one label subset for this species.
            2) Generate a model from the sampled markdown features.
            3) Generate a trigger script for the model.
            4) Enhance the trigger script (AST-based stabilization).
            5) Optionally classify the final enhanced script as [normal] / [abnormal].

    Key differences from the previous version:
      - No crash/inconsistency split in the directory structure.
      - Each sampling (for a given species) goes through the full pipeline
        (model -> trigger -> enhance -> classify) before moving to the next species.
      - The output structure under `output_root/<timestamp>/` is:

            models/
                <species_basename>/*.py
            bugs_trigger/
                <species_basename>/*.py
                <species_basename>/normal/*.py/.md
                <species_basename>/abnormal/*.py/.md

    Notes:
        - `cluster_md_directory` is the root that contains cluster markdowns such as
          "facebook-incoder-1B-dbscan-cluster_results-0.md" and their optional
          trigger counterparts "*-trigger.md".
        - `issue_csv_root` should point to the directory that contains per-species
          CSV files mapping original issues to cluster labels; by default it is
          the same as `cluster_md_directory`.
    """
    if issue_csv_root is None:
        issue_csv_root = cluster_md_directory

    start_time = datetime.now()
    deadline = start_time + timedelta(hours=max_hours)

    # Map from "<cluster_md_directory>/<species_basename>" -> max_label_index
    cluster_result = find_max_version(cluster_md_directory)
    if not cluster_result:
        print(f"[WARN] No cluster markdown files found under: {cluster_md_directory}")
        return
    # Global classification roots (siblings of each timestamp directory):
    #   output_root/normal/<species>/...
    #   output_root/abnormal/<species>/...
    global_normal_root = os.path.join(output_root, "normal")
    global_abnormal_root = os.path.join(output_root, "abnormal")
    os.makedirs(global_normal_root, exist_ok=True)
    os.makedirs(global_abnormal_root, exist_ok=True)


    species_list = sorted(cluster_result.keys())
    print(f"[INFO] Detected {len(species_list)} species (encoding schemes) from clustered markdowns.")

    for round_idx in range(num_rounds):
        print(f"==================== Round {round_idx + 1}/{num_rounds} ====================")
        print(f"[INFO] Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if datetime.now() >= deadline:
            print("[INFO] Reached max running time; exiting before starting new round.")
            break

        # Each round has its own timestamped root directory
        timestamp = time.strftime("%Y%m%d%H%M%S")
        result_root = os.path.join(output_root, timestamp)
        models_root = os.path.join(result_root, "models")
        triggers_root = os.path.join(result_root, "bugs_trigger")
        os.makedirs(models_root, exist_ok=True)
        os.makedirs(triggers_root, exist_ok=True)

        for species in species_list:
            if datetime.now() >= deadline:
                print("[INFO] Reached max running time during species loop; exiting.")
                break

            max_label_index = cluster_result[species]
            species_basename = os.path.basename(species)

            # Determine how many labels to sample for this species
            # max_quantity cannot exceed (max_label_index + 1)
            max_quantity = min(label_max, max_label_index + 1)
            if label_min > max_quantity:
                print(
                    f"[WARN] label_min={label_min} is greater than available labels "
                    f"({max_label_index + 1}) for species '{species_basename}'. Skip."
                )
                continue

            quantity = random.randint(label_min, max_quantity)

            print(
                f"[INFO] Round {round_idx + 1}: sampling species '{species_basename}' "
                f"with quantity={quantity} (labels in [0, {max_label_index}])."
            )

            # ------------------------------------------------------------------
            # 1) Generate model for this species and this sampling
            # ------------------------------------------------------------------
            try:
                md_str, label_list = _generate_md_from_labels(
                    species=species,
                    quantity=quantity,
                    versions=cluster_result,
                    cluster_md_dir=cluster_md_directory,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to generate markdown for species '{species_basename}': {exc}")
                continue

            model_question = md_str

            model_answer = ask_gpt_langchain(
                question=model_question,
                model=llm_model,
                api_key=api_key,
                api_base=llm_api_base,
                prompt_file=model_prompt_file,
                temperature=llm_temperature,
            )

            if not model_answer:
                print(f"[WARN] Empty model answer for species '{species_basename}', skip this sample.")
                continue

            model_source = extract_python_code(model_answer)
            model_path = save_string_to_python_file(
                base_dir=models_root,
                sub_dir=species_basename,
                content=model_source,
                labels=label_list,
            )
            print(f"[INFO] Generated model file: {model_path}")
            log_elapsed_time(start_time, prefix="[INFO] Elapsed after model generation")

            if datetime.now() >= deadline:
                print("[INFO] Reached max running time after model generation; exiting.")
                break

            # ------------------------------------------------------------------
            # 2) Generate trigger script(s) for the freshly generated model
            # ------------------------------------------------------------------
            try:
                model_folder, labels_for_trigger, file_name, _ = parse_python_file_path(
                    file_path=model_path,
                    root_dir=models_root,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to parse model filename '{model_path}': {exc}")
                continue

            csv_path = os.path.join(issue_csv_root, f"{model_folder}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARN] CSV file not found for model_folder '{model_folder}': {csv_path}")
                continue

            try:
                code_block = _generate_code_block_from_labels(
                    label_list=labels_for_trigger,
                    csv_path=csv_path,
                    trigger_md_dir=cluster_md_directory,
                    species=model_folder,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] Failed to construct code block from labels for "
                    f"model_folder '{model_folder}': {exc}"
                )
                continue

            with open(model_path, "r", encoding="utf-8") as model_file:
                model_str = model_file.read()

            trigger_question = "This is model file:\n" + model_str + "\n" + code_block

            trigger_answer = ask_gpt_langchain(
                question=trigger_question,
                model=llm_model,
                api_key=api_key,
                api_base=llm_api_base,
                prompt_file=trigger_bug_prompt_file,
                temperature=llm_temperature,
            )

            if not trigger_answer:
                print(
                    f"[WARN] Empty trigger answer for species '{species_basename}', "
                    "skip enhancement and classification for this sample."
                )
                continue

            trigger_source = extract_python_code(trigger_answer)
            trigger_dir = os.path.join(triggers_root, model_folder)
            os.makedirs(trigger_dir, exist_ok=True)

            trigger_path = save_code_to_timestamped_file(
                content=trigger_source,
                name_part=file_name,
                output_dir=trigger_dir,
            )
            print(f"[INFO] Generated trigger file: {trigger_path}")
            log_elapsed_time(start_time, prefix="[INFO] Elapsed after trigger generation")

            if datetime.now() >= deadline:
                print("[INFO] Reached max running time after trigger generation; exiting.")
                break

            # ------------------------------------------------------------------
            # 3) Enhance the trigger script (single-file AST stabilization)
            # ------------------------------------------------------------------
            enhanced_path: Optional[str] = trigger_path
            for iteration in range(max_iterations_per_file):
                if enhanced_path is None:
                    break

                try:
                    with open(enhanced_path, "r", encoding="utf-8") as before_file:
                        tree_before = ast.parse(before_file.read())
                except Exception as exc:  # noqa: BLE001
                    tree_before = None
                    print(
                        f"[WARN] AST parse error (before enhancement) for "
                        f"'{enhanced_path}': {exc}"
                    )

                new_path = _enhance_single_file(
                    root_dir=triggers_root,
                    file_path=enhanced_path,
                    prompt_file=enhance_prompt_file,
                    api_key=api_key,
                    model=llm_model,
                    api_base=llm_api_base,
                    temperature=llm_temperature,
                )

                if new_path is None:
                    print(
                        f"[WARN] Enhancement returned None for '{enhanced_path}', "
                        "stop enhancement for this file."
                    )
                    break

                enhanced_path = new_path

                try:
                    with open(enhanced_path, "r", encoding="utf-8") as after_file:
                        tree_after = ast.parse(after_file.read())
                except Exception as exc:  # noqa: BLE001
                    tree_after = None
                    print(
                        f"[WARN] AST parse error (after enhancement) for "
                        f"'{enhanced_path}': {exc}"
                    )

                if tree_before and tree_after and ast.dump(tree_before) == ast.dump(tree_after):
                    print(
                        f"[INFO] Trigger script stabilized after {iteration + 1} "
                        f"enhancement iterations: {enhanced_path}"
                    )
                    break

            if enhanced_path is None:
                print("[WARN] No enhanced path available; skip classification for this sample.")
                continue

            print(f"[INFO] Final enhanced trigger file: {enhanced_path}")
            log_elapsed_time(start_time, prefix="[INFO] Elapsed after enhancement")

            if datetime.now() >= deadline:
                print("[INFO] Reached max running time after enhancement; exiting.")
                break

            # ------------------------------------------------------------------
            # 4) Classify the enhanced trigger file (optional)
            # ------------------------------------------------------------------
            if enable_step5_classification:
                print("[INFO] Classifying enhanced trigger file via LLM...")
                _handle_bug_trigger_output(
                    file_path=enhanced_path,
                    prompt_file=classify_prompt_file,
                    api_key=api_key,
                    model=llm_model,
                    api_base=llm_api_base,
                    temperature=llm_temperature,
                    enable_gpu_selection=enable_gpu_selection_for_step5,
                    classify_root_normal=global_normal_root,
                    classify_root_abnormal=global_abnormal_root,
                )

                log_elapsed_time(start_time, prefix="[INFO] Elapsed after classification")

                # A second-pass consistency check is not strictly necessary here
                # because we classify per-file. You may still call move_abnormal_files
                # at the directory level if you extend the pipeline to bulk mode.

                log_elapsed_time(start_time, prefix="[INFO] Elapsed after classification")

            # After finishing the full pipeline for this species/sample, continue
            # with the next species (or break if time is over).

        # End of species loop for this round

        if datetime.now() >= deadline:
            print("[INFO] Reached max running time at the end of the round; exiting.")
            break

    end_time = datetime.now()
    print(f"[INFO] Total wall-clock time: {_format_timedelta(start_time, end_time)}")
