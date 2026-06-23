# NMR Input-Modality Ablation Design

## Objective

Measure the independent and complementary contributions of spectrum images,
tabulated peaks, and molecular formula to direct canonical-SMILES prediction.

## Conditions

`full` provides two ordered 1H/13C images, both peak tables, and formula.
`image_only` provides the two images and formula. `peak_table_only` provides
both peak tables and formula without images. `formula_only` provides only the
formula and measures dataset/molecular-prior leakage. All conditions use the
same scaffold-disjoint IDs, targets, optimizer settings, prompt index, seed,
early stopping, and evaluation metrics.

## Implementation

A validated `input_mode` is shared by prompt construction, lazy dataset
loading, training logs, inference input assembly, and prediction records.
Text-only modes do not read or render images. Rules and auxiliary tasks are
disallowed in non-full modes to prevent hidden spectral evidence. The initial
matrix uses `scale-5k` as the 5k full condition and adds three 5k ablations;
50k adds image-only and peak-table-only against `main-3407` as the full model.
