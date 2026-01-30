# HBTGen

A command-line pipeline for:

* encoding issue code into vectors
* clustering and feature generation
* LLM-driven script evaluation loops

The tool supports multiple run modes and compilers and is designed to be easily extensible.

---

## Installation

### 1. Clone repository

```bash
git clone <your-repo-url>
cd <repo>
```

### 2. Create environment

Python 3.9+ is recommended.

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure (Simplified)

```
.
├── data/
│   ├── pytorch-issue/
│   ├── tensorflow-issue/
│   └── ...
├── src/
│   ├── abstract/
│   ├── cluster_and_feature_generate/
│   └── generate_scripts/
├── llm_config.json
└── hbtgen_main.py
```

---

## LLM Configuration

Feature and script modes require an LLM config file:

Create `llm_config.json` in the working directory:

```json
{
  "MODEL": "your-llm-model",
  "API_KEY": "your-api-key",
  "API_BASE": "https://api.example.com",
  "TEMPERATURE": 0.6
}
```

---

## Usage

All commands are run through:

```bash
python hbtgen_main.py [OPTIONS]
```

### Common Options

| Option         | Description                                  |
| -------------- | -------------------------------------------- |
| `--mode`       | Run mode: `abstract`, `feature`, or `script` |
| `--compiler`   | Target compiler: `pytorch` or `tensorflow`   |
| `--encoder-id` | Encoder model ID (required in abstract mode) |

---

## Mode: ABSTRACT

Encodes issue code into vectors and performs clustering.

*Note:* The default encoding method is `umap`; you can modify it in the source code according to your needs.

```python
reduction="umap",
```

### Required arguments

* `--mode abstract`
* `--compiler`
* `--encoder-id`

### Example

```bash
python hbtgen_main.py \
  --mode abstract \
  --compiler pytorch \
  --encoder-id 5
```

Encoder IDs:

```
1: microsoft/codebert-base
2: Salesforce/codet5-base
3: facebook/incoder-1B
4: microsoft/graphcodebert-base
5: Qwen/Qwen2.5-7B-Instruct
6: codellama/CodeLlama-7b-hf
```

---

## Mode: FEATURE

Generates feature markdowns from clustered CSV files using an LLM.

### Required arguments

* `--mode feature`
* `--compiler`

### Example

```bash
python hbtgen_main.py \
  --mode feature \
  --compiler pytorch
```

Requires a valid `llm_config.json`.

---

## Mode: SCRIPT

Runs an evaluation loop driven by LLM prompts.

### Required arguments

* `--mode script`
* `--compiler`
* `--time` (hours per round)
* `--round` (number of rounds)

### Example

```bash
python hbtgen_main.py \
  --mode script \
  --compiler pytorch \
  --time 4 \
  --round 3
```

This runs 3 rounds, each capped at 4 hours.

---

## Notes

* Output artifacts are written under `./data/`
* Script outputs are written under `./generate/`
* GPU is auto-selected when available

