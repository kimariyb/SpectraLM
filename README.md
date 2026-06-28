# SpectraLM

SpectraLM is a text-only research codebase for small-molecule structure
prediction from paired one-dimensional `1H` and `13C` NMR peak tables. The
current project no longer uses spectrum images or image-based model training;
the active question is whether a Qwen3-8B text LLM can learn useful
NMR-to-structure mappings, especially when an explicit molecular formula is
available.

## Current Research State

The active architecture is:

```text
1H peak table + 13C peak table + optional molecular formula
                  |
                  v
            Qwen3-8B text LLM
                  |
                  v
       direct JSON SMILES / Top-k candidates
                  |
                  v
 canonicalization + formula/domain hard filters
                  |
                  v
       1D NMR rules + candidate reranking
```

The model is trained to return exactly one JSON object:

```json
{"smiles":"string (canonical SMILES, or null if insufficient data)"}
```

The prompt, peak tables, formula, and optional rule context are input context.
Training uses response-only supervision so the loss is applied only to the
assistant target, not to the long NMR prompt.

## Main Findings So Far

The latest completed pilot is the controlled 10k split:

- train: 8,000 molecules
- validation: 1,000 molecules
- test: 1,000 molecules
- model: `/mnt/data/kimariyb/models/Qwen3-8B`
- target: connectivity canonical SMILES with stereochemistry removed
- molecule domain: single neutral, non-radical, isotope-free component
  containing only H, C, N, O, F, Si, P, S, Cl, Br, or I

Results currently stored in `outputs/experiments/structure/predictions/` show:

| Experiment | Exact | Valid SMILES | Formula accuracy | Mean Tanimoto | Functional-group micro F1 |
|---|---:|---:|---:|---:|---:|
| `direct-formula-10k` | 0.2% | 94.3% | 2.6% | 0.183 | 0.657 |
| `direct-no-formula-10k` | 0.0% | 95.7% | 0.6% | 0.175 | 0.588 |
| `candidates-formula-10k` | 0.4% | 50.0% | 50.0% | 0.107 | 0.468 |
| `candidates-no-formula-10k` | 0.0% | 100.0% | 0.3% | 0.178 | 0.615 |

The current interpretation is:

- The text LLM has learned to produce chemically valid SMILES and some local
  NMR/functional-group associations.
- Formula-conditioned training is useful, but prompt-only formula conditioning
  is not enough; direct generation usually violates the formula.
- Exact structure recovery from 1D NMR is still very weak in the current pilot.
  Most failures are plausible-looking molecules with incorrect connectivity,
  ring size, heteroatom placement, or substituent arrangement.
- Current candidate ranking is limited by candidate recall. The top-32
  generated candidate set almost never contains the reference structure, so the
  reranker cannot yet recover the correct answer.
- The rule library has not yet been used as training context in the reported
  runs (`rule_context_enabled: false`), so it should currently be described as
  a post-generation validation/reranking component, not as a demonstrated
  training improvement.

This means the project should not claim that direct single-shot LLM generation
solves NMR structure elucidation. The defensible research direction is a
constraint-aware text-LLM system: generate multiple candidate structures,
canonicalize them, enforce domain and formula constraints, and then rerank with
NMR rules plus model evidence.

## Supported Workflow

Activate the CUDA training environment first:

```bash
conda activate ml
```

Build the full paired JSONL dataset from the raw CSV:

```bash
bash script/build_full_jsonl.sh \
  dataset/NMRexp_10to24_1_1004.csv \
  dataset/paired_jsonl_full
```

Create the controlled 10k split and formula-matched candidate sidecars:

```bash
bash script/run_experiment.sh prepare split-10k
bash script/run_experiment.sh prepare candidates-formula-10k-train
bash script/run_experiment.sh prepare candidates-formula-10k-val
```

Run a smoke training job:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train smoke
```

Train the formula-conditioned path:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-formula-10k
```

Train the no-formula ablation:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage1-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh train stage2-no-formula-10k
```

Evaluate direct generation and candidate generation:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer direct-no-formula-10k
CUDA_VISIBLE_DEVICES=0 bash script/run_experiment.sh infer candidates-no-formula-10k
```

## Repository Layout

- `src/data/`: molecule normalization, dataset loading, task construction,
  functional-group labels, and spectral-region labels.
- `src/evaluation/`: prompts, structure metrics, generation diagnostics, and
  spectral-consistency metrics.
- `src/nmr_rules/`: deterministic 1D NMR evidence generation and candidate
  validation.
- `src/training/`: text-only QLoRA training, direct inference, constrained
  candidate inference, and response-only collation.
- `script/`: supported dataset preparation and experiment runners.
- `configs/`: one smoke config plus the active 10k formula/no-formula training
  and inference configs.
- `rules/nmr_1d.yaml`: machine-readable 1D NMR rule library.
- `docs/experiments.md`: exact experiment command protocol.
- `docs/research_design.md`: research design, metrics, and next-step logic.
- `docs/nmr_1d_rulebook.md`: human-readable 1D NMR interpretation rulebook.

## Key Metrics

Primary reporting should use connectivity exact match. Secondary metrics are:

- exact canonical SMILES match
- valid SMILES rate
- molecular-formula accuracy
- Tanimoto similarity
- ring-scaffold match
- functional-group F1
- functional-group spectral support
- output-format compliance
- invalid-structure rate
- non-SMILES or non-JSON output rate
- candidate oracle@k
- formula-constraint failure rate
- ranking failure rate

For the current pilot, candidate oracle@32 is the most important diagnostic.
If the reference structure is absent from the generated candidate set, no
reranking method can recover it.

## Near-Term Research Steps

1. Improve candidate generation so formula-valid candidate recall increases
   before investing more effort in reranking.
2. Enforce molecular formula as a hard post-generation constraint for all
   formula-conditioned inference.
3. Report direct, candidate, formula, and no-formula results separately.
4. Add a rule-context ablation only after the baseline candidate workflow is
   stable.
5. Scale beyond 10k only after candidate oracle@k and formula accuracy improve.

## Verification

```bash
conda activate ml
python -m compileall src tests script
pytest
```
