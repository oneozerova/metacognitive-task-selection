# Metacognitive Curriculum Selection Experiment

**Research question:** Does metacognitive task selection improve an LLM agent's performance on held-out programming tasks compared to random or fixed curricula?

Minimal, reproducible study of *intrinsic metacognitive learning* — Liu & van der Schaar (ICML 2025).  
Tasks sourced from **BigCodeBench** (ICLR 2025): real-world Python tasks requiring diverse stdlib function calls.

---

## Results (gpt-oss-20b, 3 seeds × 3 conditions, 7 eval tasks)

| Condition | Mean Initial Acc | Mean Final Acc | Mean Learning Gain |
|---|---|---|---|
| `random` | 0.809 | 0.809 | **+0.000** |
| `metacognitive` | 0.809 | 0.762 | **−0.048** |
| `fixed` | 0.809 | 0.714 | **−0.095** |

The central hypothesis — metacognitive > random — is **not supported**. All observed gains are within ±1 task (±14.3%) and indistinguishable from noise given the 7-task eval resolution.

**Why it fails:** The task set has bimodal difficulty for this model. Three eval tasks fail 0% of the time regardless of practice (`file_ops_eval_01/02`, `regex_eval_02`); six tasks pass 100% of the time. Only one task sits in the model's zone of proximal development, leaving no meaningful signal for any curriculum to exploit. The LLM makes intelligent selections (task selection quality = 0.71) but calibration error averages 0.40, and there is nothing for it to actually learn.

---

## Experimental design

Three conditions, same model, same number of practice and eval tasks:

| Condition | Task selection | Purpose |
|---|---|---|
| `random` | Uniform random | Weak baseline |
| `fixed` | `regex → text_transform → json_proc → math_compute → data_struct → file_ops` | Human-designed curriculum baseline |
| `metacognitive` | LLM self-selects by assessing per-skill weaknesses and expected learning gain | Main hypothesis |

Learning is in-context only: after each failed practice task the model writes a compact rule injected into all subsequent prompts (max 12 rules). No fine-tuning or weight updates.

---

## Project structure

```
metacognitive-task-selection/
├── data/
│   ├── tasks_practice_bigcodebench.json   # 36 practice tasks (6 per skill × 6 skills)
│   ├── tasks_eval_bigcodebench.json       # 12 held-out eval tasks (2 per skill); 7 used at runtime by default
│   └── fetch_bigcodebench.py              # regenerate task files from HuggingFace
├── prompts/
│   ├── solve_prompt.txt
│   ├── select_task_prompt.txt
│   └── reflection_prompt.txt
├── src/
│   ├── run_experiment.py     # entry point
│   ├── llm_client.py         # Yandex AI Studio wrapper (OpenAI-compatible)
│   ├── evaluator.py          # subprocess runner; structured + BigCodeBench unittest harness
│   ├── curriculum.py         # random / fixed / metacognitive selection strategies
│   └── metrics.py            # metrics computation + matplotlib plots
├── outputs/                  # created at runtime (gitignored)
│   ├── logs.jsonl
│   ├── results.csv
│   └── plots/
└── docs/self-improving-agents.pdf
```

---

## Setup

```bash
pip install -r requirements.txt
export YANDEX_API_KEY="your-key"
```

Override the model (default: `gpt://b1gabcde1234/gpt-oss-20b`):
```bash
export EXPERIMENT_MODEL="gpt://YOUR_FOLDER_ID/model-name"
```

---

## Running

```bash
# Full experiment — 3 conditions × 3 seeds
python src/run_experiment.py

# Single condition, all seeds
python src/run_experiment.py --conditions metacognitive --seeds 1 2 3

# Quick smoke test
python src/run_experiment.py --conditions random --seeds 1 --num-practice 1 --num-eval 6

# Regenerate plots from existing logs
python src/run_experiment.py --analyze
```

Run from the **project root**. Results are saved incrementally to `outputs/results.csv` after each completed run — a crash does not lose prior results.

### Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--num-practice` | `7` | Practice tasks per run |
| `--num-eval` | `7` | Eval tasks; selected round-robin by skill to ensure all 6 skills are covered before any skill repeats |
| `--seeds` | `1 2 3` | Multiple seeds give variance estimates for error bars |

---

## Task domain

Six BigCodeBench skill categories (stdlib-only, no external packages required):

| Skill | Primary modules |
|---|---|
| `regex` | `re` |
| `json_proc` | `json`, `csv` |
| `text_transform` | `string`, `textwrap`, `difflib`, `unicodedata` |
| `math_compute` | `math`, `statistics`, `decimal`, `fractions` |
| `data_struct` | `collections`, `itertools`, `functools`, `heapq` |
| `file_ops` | `os`, `pathlib`, `shutil`, `tempfile`, `glob`, `zipfile` |

Each task is evaluated via `unittest.TestCase` — 5–9 test methods run in a subprocess with a 10 s timeout.

---

## Metrics

| Metric | Description |
|---|---|
| `initial_accuracy` | Eval accuracy before practice |
| `final_accuracy` | Eval accuracy after practice |
| `learning_gain` | `final − initial` **(main metric)** |
| `practice_accuracy` | Accuracy during the practice loop |
| `calibration_error` | `mean(|estimated_competence − actual_rate|)` — metacognitive only |
| `task_selection_quality` | Fraction of selections that targeted a genuinely weak skill — metacognitive only |

Plots saved to `outputs/plots/`: accuracy by condition, learning gain with error bars, skill-wise gain, calibration scatter.

---

## Reference

Liu, T. & van der Schaar, M. (2025). *Truly Self-Improving Agents Require Intrinsic Metacognitive Learning.* ICML 2025. arXiv:2506.05109

Zhuo, T. Y. et al. (2025). *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions.* ICLR 2025. arXiv:2406.15877
