#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangChain LLM Client Utilities

This module exposes a single public callable:
    - ask_gpt_langchain(...)

Key features:
- LangChain ChatOpenAI invocation with message chunking for long prompts.
- Retry with exponential backoff (tenacity).
- Token-aware chunking using tiktoken when available, with a safe fallback.
- Optional external prompt file (e.g., prompt.md) injected as the final task instruction.
- No side effects at import time (no proxy/env mutations, no hard-coded keys).

Notes:
- The caller is responsible for supplying model/api_key/api_base and any proxy configuration.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


MODEL="QwQ-32B"
API_KEY="EMPTY"
API_BASE="http://114.212.170.115:11288/v1"
TEMPERATURE=0.6

# -----------------------------
# File utilities
# -----------------------------

def _read_text_file(file_path: str) -> str:
    """
    Read a text file and return its content as a string.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# Chunking utilities
# -----------------------------

def _try_count_tokens(text: str, model: str) -> Optional[int]:
    """
    Best-effort token counting using tiktoken.
    Returns None if tiktoken is unavailable.
    """
    try:
        import tiktoken  # local import to keep dependency optional
    except Exception:
        return None

    # Most OpenAI-compatible models can be approximated using cl100k_base.
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, disallowed_special=()))


def _split_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks each containing at most max_tokens tokens (tiktoken).
    Falls back to a single chunk if tiktoken is not available.

    This function assumes cl100k_base encoding which is commonly used for OpenAI-style models.
    """
    try:
        import tiktoken  # local import
    except Exception:
        # Fallback: return whole text, the caller can do char-splitting instead.
        return [text]

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())

    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
        i += max_tokens
    return chunks


def _split_text_by_chars(text: str, chunk_size: int) -> List[str]:
    """Simple character-based chunking fallback."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _chunk_question(
    question: str,
    model: str,
    max_tokens_per_chunk: int,
    fallback_char_chunk_size: int,
) -> List[str]:
    """
    Token-aware chunking when possible; otherwise fallback to char-based chunking.
    """
    token_count = _try_count_tokens(question, model)
    if token_count is None:
        return _split_text_by_chars(question, fallback_char_chunk_size)

    if token_count <= max_tokens_per_chunk:
        return [question]

    token_chunks = _split_text_by_tokens(question, max_tokens=max_tokens_per_chunk)
    # If token chunking fails and returns a single huge chunk, fallback to chars.
    if len(token_chunks) == 1 and token_count > max_tokens_per_chunk:
        return _split_text_by_chars(question, fallback_char_chunk_size)
    return token_chunks


# -----------------------------
# Message construction
# -----------------------------

def _build_messages_for_chunked_prompt(
    chunks: List[str],
    sys_prompt: str,
    final_instruction: str,
) -> List:
    """
    Build LangChain message list:
    - System prompt
    - Intro instruction
    - Chunk messages (numbered)
    - Final instruction to execute the task
    """
    if not chunks:
        raise ValueError("chunks must not be empty")

    messages = [SystemMessage(content=sys_prompt)]

    messages.append(
        HumanMessage(
            content=(
                f"The following input consists of {len(chunks)} chunk(s). "
                "Please read them in order and use all of them to answer the final instruction."
            )
        )
    )

    for idx, chunk in enumerate(chunks, 1):
        messages.append(HumanMessage(content=f"[Chunk {idx}/{len(chunks)}]\n{chunk}"))

    messages.append(HumanMessage(content=final_instruction))
    return messages


# -----------------------------
# Public API
# -----------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(Exception),
)
def ask_gpt_langchain(
    question: str,
    model: str,
    api_key: str,
    api_base: str,
    *,
    # Role / behavior
    sys_prompt: str = "You are an expert in deep learning and software engineering.",
    # Optional external prompt file (e.g., a long task description / requirements).
    prompt_file: Optional[str] = None,
    # If provided, overrides the automatically constructed final instruction.
    final_instruction: Optional[str] = None,
    # Sampling / runtime configuration
    temperature: float = 0.6,
    max_tokens_per_chunk: int = 3000,
    fallback_char_chunk_size: int = 3000,
    max_retries: int = 10,
    request_timeout: Optional[float] = None,
) -> str:
    """
    LangChain-compatible LLM call with robust chunking + retry and optional prompt file.

    Args:
        question:
            Full user prompt (e.g., concatenated code files). If too long, it will be chunked.
        model:
            Model name (OpenAI-compatible), e.g. "gpt-4o", "QwQ-32B", etc.
        api_key:
            API key (provided by caller).
        api_base:
            Base URL for OpenAI-compatible endpoint (provided by caller).
        sys_prompt:
            System prompt injected as the first message (role + general behavior).
        prompt_file:
            Path to an external prompt file (e.g., a Markdown file describing the task).
            If provided and `final_instruction` is None, the file content will be used as
            the final task instruction.
        final_instruction:
            Explicit final instruction message. If not None, it overrides any `prompt_file`.
        temperature:
            Sampling temperature.
        max_tokens_per_chunk:
            Token-based chunk size (requires tiktoken). If tiktoken is unavailable, it
            is ignored and character-based chunking is used instead.
        fallback_char_chunk_size:
            Character-based chunk size used when token counting/splitting is unavailable.
        max_retries:
            Retry count passed to ChatOpenAI itself (in addition to tenacity wrapper).
        request_timeout:
            Optional per-request timeout (seconds) if supported by the underlying client.

    Returns:
        Assistant response text.

    Notes:
        - This function does not mutate environment variables or proxy settings.
        - `prompt_file` is typically where you put a long, structured task description,
          such as "Given all provided PyTorch model files from one cluster, extract a
          modular template with the following requirements: ...".
        - Retry is handled by tenacity (outer) and ChatOpenAI (inner).
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("api_base must be a non-empty string")

    # Load external prompt file if provided and no explicit final_instruction is given
    prompt_text: Optional[str] = None
    if prompt_file is not None:
        prompt_text = _read_text_file(prompt_file)

    if final_instruction is None:
        if prompt_text is not None:
            # Default pattern: code chunks are the information, prompt file is the task spec.
            final_instruction = (
                "All chunks above contain the full input context. "
                "Now execute the following task based on all of them:\n\n"
                f"{prompt_text}"
            )
        else:
            final_instruction = (
                "All chunks have been provided above. "
                "Please now execute the task based on all the information."
            )

    chunks = _chunk_question(
        question=question,
        model=model,
        max_tokens_per_chunk=max_tokens_per_chunk,
        fallback_char_chunk_size=fallback_char_chunk_size,
    )

    messages = _build_messages_for_chunked_prompt(
        chunks=chunks,
        sys_prompt=sys_prompt,
        final_instruction=final_instruction,
    )

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        openai_api_key=api_key,
        openai_api_base=api_base,
        request_timeout=request_timeout,
    )

    response = llm.invoke(messages)
    return getattr(response, "content", str(response))
