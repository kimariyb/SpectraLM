# Dataset Molecule Policy Design

## Scope

The paired NMR dataset accepts only one sanitized molecular component built
from `H C N O F Si P S Cl Br I`, with zero net formal charge and no radical
electrons. Neutral charge-separated representations, such as nitro groups,
remain valid. Isotope labels are removed and the molecule is canonicalized
again using natural-abundance atoms.

## Data Flow

Raw SMILES are inspected before insertion into the candidate SQLite index.
Rejected rows are counted by reason; accepted isotope-labelled rows are
indexed under their isotope-free canonical SMILES so proton and carbon records
coalesce correctly. Pair construction repeats the policy check for old reused
indexes, and manifest construction records structural QC fields and reasons.

## Validation

Tests cover disconnected salts, non-zero net charge, radicals, isotope
normalization, neutral charge-separated molecules, unsupported elements, index
statistics, and paired JSONL output.
