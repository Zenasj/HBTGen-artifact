#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI entrypoint for the pipeline.

Modes:
  - abstract: encode issue code into vectors + run clustering.
  - feature : generate cluster-level features using an LLM config.
  - script  : run an LLM-driven evaluation loop for multiple rounds.

This file is intentionally structured to be easy to extend:
  - Add a new mode in RunMode + argparse choices + parse_args_to_config() + run_with_config().
  - Add a new compiler in TargetCompiler + argparse choices + parse_args_to_config().
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.abstract.abstract import build_and_save_code_vectors
from src.cluster_and_feature_generate.cluster import run_clustering_for_vector_directories

ROOT = Path(__file__).resolve().parent

# LLM config keys expected in llm_config.json (working directory by default).
REQUIRED_FIELDS = ["MODEL", "API_KEY", "API_BASE", "TEMPERATURE"]


def load_llm_config(filename: str = "llm_config.json") -> dict:
    """
    Load LLM configuration from a JSON file in the current working directory.

    Expected schema:
      {
        "MODEL": "...",
        "API_KEY": "...",
        "API_BASE": "...",
        "TEMPERATURE": 0.0
      }
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Config file '{filename}' not found in the current directory."
        )

    with open(filename, "r", encoding="utf-8") as f:
        config = json.load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # Normalize keys and types for downstream code.
    return {
        "model": config["MODEL"],
        "api_key": config["API_KEY"],
        "api_base": config["API_BASE"],
        "temperature": float(config["TEMPERATURE"]),
    }


# ==== 1) Extensible enums ====


class RunMode(Enum):
    ABSTRACT = auto()
    FEATURE = auto()
    SCRIPT = auto()
    # Add more modes here (and update argparse + parse_args_to_config + run_with_config).


class TargetCompiler(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    # Add more compilers here (e.g., TVM = "tvm").


# ==== 2) Encoder model registry ====

ENCODER_MODELS = [
    "microsoft/codebert-base",        # 1
    "Salesforce/codet5-base",         # 2
    "facebook/incoder-1B",            # 3
    "microsoft/graphcodebert-base",   # 4
    "Qwen/Qwen2.5-7B-Instruct",       # 5
    "codellama/CodeLlama-7b-hf",      # 6
]

# Use 1-based indices for convenient CLI input.
ENCODER_ID2NAME = {idx + 1: name for idx, name in enumerate(ENCODER_MODELS)}


# ==== 3) Parsed configuration ====

@dataclass
class RunConfig:
    mode: RunMode
    compiler: TargetCompiler

    # For abstract mode: encoder_id/name must be provided.
    # For feature/script modes: encoder_id/name must be None.
    encoder_id: Optional[int] = None
    encoder_name: Optional[str] = None

    # For script mode only.
    time_hours: Optional[float] = None
    num_rounds: Optional[int] = None


# ==== 4) CLI parsing ====


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pipeline with different modes/encoders/compilers."
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="abstract",
        choices=["abstract", "feature", "script"],
        help="Run mode: abstract / feature / script",
    )

    # Encoder ID is only required in abstract mode (validated in parse_args_to_config()).
    parser.add_argument(
        "--encoder-id",
        "-e",
        type=int,
        required=False,
        choices=list(ENCODER_ID2NAME.keys()),
        help=(
            "Encoder model ID (required for --mode abstract):\n"
            "  1: microsoft/codebert-base\n"
            "  2: Salesforce/codet5-base\n"
            "  3: facebook/incoder-1B\n"
            "  4: microsoft/graphcodebert-base\n"
            "  5: Qwen/Qwen2.5-7B-Instruct\n"
            "  6: codellama/CodeLlama-7b-hf"
        ),
    )

    # Compiler is required for all modes.
    parser.add_argument(
        "--compiler",
        "-c",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow"],
        help="Target compiler: pytorch / tensorflow",
    )

    # Script mode only: max hours per round.
    parser.add_argument(
        "--time",
        type=float,
        required=False,
        help="(script mode) Max time per round, in hours. Example: 4.0",
    )

    # Script mode only: number of rounds.
    parser.add_argument(
        "--round",
        dest="num_rounds",
        type=int,
        required=False,
        help="(script mode) Number of rounds (positive integer).",
    )

    return parser


def parse_args_to_config() -> RunConfig:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parse compiler.
    if args.compiler == "pytorch":
        compiler = TargetCompiler.PYTORCH
    elif args.compiler == "tensorflow":
        compiler = TargetCompiler.TENSORFLOW
    else:
        raise ValueError(f"Unsupported compiler: {args.compiler}")

    # Parse mode + enforce per-mode constraints.
    if args.mode == "abstract":
        mode = RunMode.ABSTRACT

        # Abstract mode requires encoder-id.
        if args.encoder_id is None:
            parser.error("--encoder-id is required when --mode abstract")

        encoder_id = args.encoder_id
        encoder_name = ENCODER_ID2NAME[encoder_id]

        # Not used in abstract mode.
        time_hours = None
        num_rounds = None

    elif args.mode == "feature":
        mode = RunMode.FEATURE

        # Feature mode should not take encoder-id.
        if args.encoder_id is not None:
            parser.error(
                "--encoder-id should NOT be provided when --mode feature "
                "(feature mode only requires --compiler)."
            )

        encoder_id = None
        encoder_name = None
        time_hours = None
        num_rounds = None

    elif args.mode == "script":
        mode = RunMode.SCRIPT

        # Script mode should not take encoder-id.
        if args.encoder_id is not None:
            parser.error(
                "--encoder-id should NOT be provided when --mode script "
                "(script mode does not use encoder models)."
            )

        # Script mode requires time and round.
        if args.time is None:
            parser.error("--time is required when --mode script (unit: hours)")
        if args.num_rounds is None:
            parser.error("--round is required when --mode script")
        if args.num_rounds <= 0:
            parser.error("--round must be a positive integer")

        encoder_id = None
        encoder_name = None
        time_hours = float(args.time)
        num_rounds = int(args.num_rounds)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return RunConfig(
        mode=mode,
        compiler=compiler,
        encoder_id=encoder_id,
        encoder_name=encoder_name,
        time_hours=time_hours,
        num_rounds=num_rounds,
    )


# ==== 5) Main dispatch ====


def run_with_config(cfg: RunConfig) -> None:
    """
    Dispatch to the corresponding pipeline stage based on cfg.mode.

    Keep the per-mode code blocks small and delegate heavy logic to modules under src/.
    """
    print("=== Parsed RunConfig ===")
    print(f"  mode         : {cfg.mode.name}")
    print(f"  compiler     : {cfg.compiler.value}")
    print(f"  encoder_id   : {cfg.encoder_id}")
    print(f"  encoder_name : {cfg.encoder_name}")
    print(f"  time_hours   : {cfg.time_hours}")
    print(f"  num_rounds   : {cfg.num_rounds}")
    print("========================")

    # Convention: outputs are stored under ./data/<compiler>-issue-abstract-result
    directory_list = [
        ROOT / "data" / f"{cfg.compiler.value}-issue-abstract-result"
    ]

    if cfg.mode == RunMode.ABSTRACT:
        # 1) Encode issues into vector files (saved to disk).
        issue_data_dir = ROOT / "data" / f"{cfg.compiler.value}-issue"

        results = build_and_save_code_vectors(
            root_dir=issue_data_dir,
            model_names=[cfg.encoder_name],
            save=True,
            vector_file_suffix="-vectors.pth",
            extract_mode="model_classes",  # or "full_file"
            device=None,  # auto: cuda if available else cpu
        )
        print(results)

        # 2) Cluster vectors and (optionally) save plots/artifacts.
        results = run_clustering_for_vector_directories(
            directory_list=directory_list,
            vector_file_pattern="*-vectors.pth",
            method="dbscan",
            reduction="umap",
            n_components=3,
            dbscan_eps=0.5,
            dbscan_min_samples=3,
            save_plot=True,
            show_plot=False,
        )

        # Print a short summary for each vector file.
        for pth_path, res in results.items():
            print(f"=== Summary for {pth_path} ===")
            artifacts = getattr(res, "artifacts", None)
            print(artifacts)

    elif cfg.mode == RunMode.FEATURE:
        # Load LLM config and generate feature markdowns for cluster CSVs.
        llm_cfg = load_llm_config()
        print("=== Loaded LLM Config for FEATURE mode ===")
        print(f"  model       : {llm_cfg['model']}")
        print(f"  api_base    : {llm_cfg['api_base']}")
        print(f"  temperature : {llm_cfg['temperature']}")
        print("=========================================")

        from src.cluster_and_feature_generate.generate_feature import (
            list_csv_files,
            generate_markdowns_for_csv_list,
        )

        csv_list = list_csv_files(directory_list[0], min_max_label_threshold=0)
        prompt_md = (
            ROOT
            / "src"
            / "cluster_and_feature_generate"
            / f"generate_feature_{cfg.compiler.value}.md"
        )

        stats = generate_markdowns_for_csv_list(
            output_dir=directory_list[0],
            csv_list=csv_list,
            model=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            api_base=llm_cfg["api_base"],
            prompt_file=prompt_md,
            min_label=0,
            max_label=None,
            extract=True,
            skip_if_exists=True,
            overwrite=False,
            enable_trigger_markdown=True,
            skip_trigger_if_exists=True,
        )
        print(stats)

    elif cfg.mode == RunMode.SCRIPT:
        # Load LLM config and run the evaluation loop for multiple rounds.
        llm_cfg = load_llm_config()
        print("=== Loaded LLM Config for SCRIPT mode ===")
        print(f"  model       : {llm_cfg['model']}")
        print(f"  api_base    : {llm_cfg['api_base']}")
        print(f"  temperature : {llm_cfg['temperature']}")
        print(f"  time_hours  : {cfg.time_hours}")
        print(f"  num_rounds  : {cfg.num_rounds}")
        print("========================================")

        from src.generate_scripts.generate_scripts import run_evaluation_loop

        script_output_dir = ROOT / "generate" / f"{cfg.compiler.value}"
        generate_model_prompt = ROOT / "src" / "generate_scripts" / f"generate_model_{cfg.compiler.value}.md"
        generate_trigger_prompt = ROOT / "src" / "generate_scripts" / f"generate_trigger_{cfg.compiler.value}.md"
        generate_enhancement_prompt = ROOT / "src" / "generate_scripts" / f"generate_enhancement_{cfg.compiler.value}.md"
        classify_prompt = ROOT / "src" / "generate_scripts" / "handle_bug_trigger_output.md"

        run_evaluation_loop(
            max_hours=cfg.time_hours,
            api_key=llm_cfg["api_key"],
            llm_model=llm_cfg["model"],
            llm_api_base=llm_cfg["api_base"],
            llm_temperature=llm_cfg["temperature"],
            cluster_md_directory=directory_list[0],
            output_root=script_output_dir,
            model_prompt_file=generate_model_prompt,
            trigger_bug_prompt_file=generate_trigger_prompt,
            enhance_prompt_file=generate_enhancement_prompt,
            classify_prompt_file=classify_prompt,
            label_min=1,
            label_max=3,
            enhance_timeout_seconds=60 * 30,
            num_rounds=cfg.num_rounds,
            enable_step5_classification=True,
            enable_gpu_selection_for_step5=True,
            unify_name_for_best_version=False,
            issue_csv_root=directory_list[0],
        )

    else:
        # Reserved for future modes.
        raise NotImplementedError(f"Logic for mode {cfg.mode} is not implemented yet.")


def main() -> None:
    cfg = parse_args_to_config()
    run_with_config(cfg)


if __name__ == "__main__":
    import os

    os.environ["http_proxy"] = "http://114.212.86.150:10809"
    os.environ["https_proxy"] = "http://114.212.86.150:10809"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["TMPDIR"] = "/data/zyzhao/tmp"
    os.environ["TEMP"] = "/data/zyzhao/tmp"
    os.environ["TMP"] = "/data/zyzhao/tmp"
    os.environ["TORCH_HOME"] = "/data/zyzhao/transformers-cache"
    os.environ["TRITON_CACHE_DIR"] = "/data/zyzhao/tmp/triton-cache"

    main()
