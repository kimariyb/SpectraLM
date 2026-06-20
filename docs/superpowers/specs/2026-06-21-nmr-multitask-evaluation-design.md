# NMR Multitask And Evaluation Design

## Objective

Extend SpectraLM with one configurable multimodal SFT model that learns direct
structure prediction plus three auxiliary tasks, while preserving the existing
single-task baselines. Restrict all dataset molecules to H, C, N, O, F, Si, P,
S, Cl, Br, and I.

## Task Protocol

Every task receives the same ordered 1H and 13C images and the same peak-table
text. Molecular formula remains optional and is read only from the explicit
input field.

| Task | Target | Default weight |
|---|---|---:|
| `structure_prediction` | One canonical SMILES | 0.60 |
| `functional_group_recognition` | Sorted JSON string array from a controlled SMARTS ontology | 0.15 |
| `candidate_ranking` | The best canonical SMILES from a displayed candidate set | 0.15 |
| `spectral_region_classification` | JSON object with sorted 1H and 13C region-label arrays | 0.10 |

Task weights are configurable, non-negative, and normalized by the data
transform. Setting structure weight to 1 and all other weights to 0 reproduces
the original training behavior.

## Candidate Data

Candidate sets are generated offline as a JSONL sidecar keyed by sample ID.
Each set contains one target and up to seven negatives with the same molecular
formula. Negatives are ordered by descending Morgan similarity to favor hard
isomeric alternatives. Sets with no formula-matched negative are omitted from
the candidate-ranking task; the transform deterministically falls back to
direct structure prediction for those samples.

The ranking target identifies only the known correct molecule. The project
does not invent an unsupported total ordering among incorrect candidates.

## Functional-Group Ontology

The first version uses presence/absence labels for common motifs: alkene,
alkyne, aromatic ring, alcohol, phenol, ether, aldehyde, ketone, carboxylic
acid, ester, amide, amine, nitrile, nitro, organohalogen, thiol, thioether,
sulfoxide, sulfone, phosphorus oxygen functionality, silicon carbon
functionality, and siloxane. SMARTS definitions are current and shared by
auxiliary targets and evaluation.

## Spectral Regions

Region targets are deterministic, multi-label summaries of observed peak
positions. They use the current soft ranges already used by the NMR rule
library. Overlapping labels are allowed. They are auxiliary labels, not ground
truth functional-group assignments.

## Evaluation

### Structure Metrics

- canonical/isomeric Exact Match;
- Valid SMILES Rate;
- Molecular Formula Accuracy;
- Morgan radius-2 Tanimoto similarity;
- Murcko Scaffold Match on reference molecules with a non-empty ring
  scaffold, plus scaffold coverage;
- Functional Group F1 from predicted and reference SMARTS label sets.

### Spectral Consistency

Functional Group Spectral Consistency checks only functional groups with a
defined 1D signature. It reports supported checks divided by applicable checks
and returns no score when no group is checkable. Chemical-shift evidence is a
soft consistency signal and is not used to redefine Exact Match.

### Model Behavior

- output-format compliance: exactly one bare, valid SMILES line;
- illegal-structure rate: one bare candidate line that RDKit cannot parse;
- non-SMILES-output rate: empty output, Markdown, labels, explanations, or
  multiple non-empty lines.

The behavior states are disjoint. A fenced but chemically valid SMILES remains
valid for structure metrics but counts as non-SMILES-formatted model output.

## Element Policy And DBE

Allowed elements are centralized as `H, C, N, O, F, Si, P, S, Cl, Br, I`.
New JSONL construction rejects other elements before writing samples. Manifest
QC records unsupported symbols. A migration command atomically filters legacy
JSONL and all split-ID files and writes excluded IDs.

For conventional common valences, DBE is:

```text
(2C + 2Si + 2 + N + P - H - F - Cl - Br - I) / 2
```

O and divalent S contribute zero. Formula-only DBE is intrinsically ambiguous
for unusual hypervalent P/S chemistry; negative or non-half-integral results
are rejected instead of silently clamped.

## Compatibility

Existing structure-only configs, prompt configurations, prediction files, and metric
keys remain readable. New metrics add fields without removing legacy aliases.
The multitask run uses a new prompt/protocol configuration and output directory.

## Verification

Tests cover element filtering, Si/P/S DBE, ontology matching, deterministic
region labels, each task target, candidate sidecar construction, task-weight
fallback, all requested metrics, behavior-state separation, configuration
validation, syntax compilation, and the full existing test suite.
