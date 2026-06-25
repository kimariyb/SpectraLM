# SpectraLM Research Design

## Positioning

SpectraLM now studies text-only molecular structure elucidation from paired
one-dimensional NMR peak tables. The core question is whether an instruction
tuned LLM can learn useful mappings from `1H`/`13C` NMR measurements and an
optional molecular formula to canonical connectivity SMILES, and whether
constraint-aware candidate generation improves direct generation.

## Active Architecture

```text
1H peak table + 13C peak table + optional formula
                  |
                  v
            Qwen3-8B text LLM
                  |
                  v
       direct SMILES / Top-k candidates
                  |
                  v
 canonicalization + optional formula hard filter
                  |
                  v
       NMR rules + candidate reranking
```

The previous image-based path is intentionally removed. Generated spectrum
images are not part of the current research claim.

## Main Claims

1. A text LLM can be instruction tuned to perform direct NMR peak-table to
   connectivity-SMILES prediction within a controlled molecular domain.
2. Molecular formula conditioning is a major information source and must be
   evaluated separately from the no-formula setting.
3. Top-k candidate generation with validity filtering, formula hard constraints,
   rule pre-ranking, and model-based reranking is a stronger system than greedy
   direct SMILES generation.
4. One-dimensional NMR is not always uniquely identifying; candidate oracle,
   constraint failure, and ranking failure are first-class outcomes.

## Data Scope

Every retained molecule must be a single neutral, non-radical, isotope-free
component containing only H, C, N, O, F, Si, P, S, Cl, Br, or I. Targets use
canonical connectivity SMILES; stereochemistry is not the primary objective.

The active 10k experiment uses:

- train: 8,000 samples
- validation: 1,000 samples
- test: 1,000 samples

## Experiments

The minimum publishable matrix is:

| Setting | Training | Inference |
|---|---|---|
| Formula-conditioned | Stage 1 multitask, Stage 2 structure | Direct and candidate reranking |
| No-formula | Stage 1 multitask, Stage 2 structure | Direct and candidate reranking |

Stage 1 includes structure prediction, functional-group recognition,
candidate ranking, and spectral-region classification. Stage 2 is
structure-only.

## Metrics

Primary metric:

- connectivity exact match

Secondary structure metrics:

- exact match
- valid SMILES rate
- molecular-formula accuracy
- Tanimoto similarity
- scaffold match
- functional-group F1

Spectral consistency:

- functional-group spectral support rate

Behavior metrics:

- output-format compliance
- invalid structure rate
- non-SMILES output rate
- candidate oracle@k
- formula constraint failure
- ranking failure

## Near-Term Steps

1. Regenerate the 8k/1k/1k split.
2. Rebuild formula-matched candidate sidecars.
3. Run smoke on one GPU.
4. Train formula-conditioned Stage 1 and Stage 2.
5. Evaluate direct and candidate inference.
6. Repeat the same path for the no-formula ablation.
7. Use the results to decide whether to scale from 10k to 50k.
