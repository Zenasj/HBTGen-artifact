#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for working with cluster-based markdown files and
LLM-generated Python code.

This module provides:

- find_max_version: discover the maximum numeric suffix for each cluster prefix.
- merge_markdown_files: concatenate multiple markdown files into a single string.
- save_string_to_python_file: persist generated Python code with label-based naming.
- extract_python_code: extract a Python code block from an LLM response and
  convert the remaining text into comments.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Iterable, List, Sequence


def find_max_version(directory: str) -> Dict[str, int]:
    """
    Scan a directory for files matching the pattern 'prefix-N.md' and return a
    mapping from the absolute prefix path to the maximum N observed.

    Example:
        If the directory contains:
            /path/to/dir/foo-0.md
            /path/to/dir/foo-1.md
            /path/to/dir/bar-3.md

        The return value will be:
            {
                "/path/to/dir/foo": 1,
                "/path/to/dir/bar": 3,
            }

    Args:
        directory: Directory to scan.

    Returns:
        A dictionary mapping "<directory>/<prefix>" to the maximum integer suffix.
    """
    pattern = re.compile(r"^(.+)-(\d+)\.md$")  # prefix-N.md
    max_versions: Dict[str, int] = {}

    try:
        entries = os.listdir(directory)
    except OSError as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to list directory: {directory}") from exc

    for filename in entries:
        match = pattern.match(filename)
        if not match:
            continue

        prefix, version_str = match.groups()
        version = int(version_str)
        key = os.path.join(directory, prefix)

        current_max = max_versions.get(key)
        if current_max is None or version > current_max:
            max_versions[key] = version

    return max_versions


def merge_markdown_files(file_list: Sequence[str]) -> str:
    """
    Read multiple markdown files and concatenate their contents into a single
    string. Each file is preceded by a header "### This is file {i}".

    Args:
        file_list: List of markdown file paths.

    Returns:
        A single string containing all file contents.
    """
    merged_content: List[str] = []

    for idx, file_path in enumerate(file_list, start=1):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except OSError as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read markdown file '{file_path}': {exc}")
            continue

        merged_content.append(f"### This is file {idx}\n\n{content}\n")

    return "\n".join(merged_content)


def save_string_to_python_file(
    base_dir: str,
    sub_dir: str,
    content: str,
    labels: Iterable[int],
) -> str:
    """
    Save a string to a Python file under `base_dir/sub_dir/` with a filename
    derived from the current timestamp and the sorted label list.

    The filename format is:
        YYYYMMDD_HHMMSS_label1_label2_... .py

    Example:
        base_dir = "out"
        sub_dir = "facebook-incoder-1B-dbscan-cluster_results"
        labels = [3, 1]

        -> out/facebook-incoder-1B-dbscan-cluster_results/
           20240218_153045_1_3.py

    Args:
        base_dir: Root directory for storing generated files.
        sub_dir: Subdirectory name under `base_dir`.
        content: Python source code to be written.
        labels: Iterable of integer labels, used to annotate the filename.

    Returns:
        The absolute path to the saved Python file.
    """
    target_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(target_dir, exist_ok=True)

    sorted_labels = "_".join(str(x) for x in sorted(labels))
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if sorted_labels:
        filename = f"{timestamp}_{sorted_labels}.py"
    else:
        filename = f"{timestamp}.py"

    file_path = os.path.join(target_dir, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"[INFO] Saved generated Python file: {file_path}")
    return file_path


def extract_python_code(response_text: str) -> str:
    """
    Extract the first Python code block from an LLM response and append any
    remaining text as comments at the end of the file.

    The function looks for a fenced code block of the form:

        ```python
        <code>
        ```

    Any non-code text (before or after the block) is converted into line
    comments prefixed with '#'.

    Args:
        response_text: Raw string returned by the LLM.

    Returns:
        A formatted Python source string that combines the extracted code and
        the remaining text as comments. If no Python code block is found, the
        entire response is returned as commented text.
    """
    # Match the first Python fenced code block
    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)

    if match:
        python_code = match.group(1).strip()
    else:
        python_code = ""

    # Split around the first ```python fence to capture non-code parts
    before_code, sep, after = response_text.partition("```python")
    if sep:
        # There was a python fence; `after` still contains code and closing ```
        code_part, sep2, after_code = after.partition("```")
        # We already used `code_part` via regex; we only care about `before_code` and `after_code`
        non_code_text = (before_code + "\n" + after_code).strip()
    else:
        # No python fence at all
        non_code_text = response_text.strip()

    comment_lines: List[str] = []
    if non_code_text:
        for line in non_code_text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            comment_lines.append(f"# {line}")

    if python_code and comment_lines:
        final_code = python_code + "\n\n" + "\n".join(comment_lines)
    elif python_code:
        final_code = python_code
    else:
        final_code = "\n".join(comment_lines)

    return final_code
