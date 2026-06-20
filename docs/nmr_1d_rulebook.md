# Practical Rulebook for 1D 1H/13C NMR Structure Elucidation

## 1. Purpose

This rulebook defines the scientific interpretation policy used by SpectraLM
for molecular structure prediction from paired one-dimensional `1H NMR` and
`13C NMR` spectra. It serves three related purposes:

1. provide a reproducible reference for human spectroscopists;
2. document the machine-readable rules in `rules/nmr_1d.yaml`;
3. define which observations may be used as model context or evaluation
   constraints without leaking the target structure.

The rules are evidence generators, not a replacement for complete structure
elucidation. Most chemical-shift ranges overlap. A chemically defensible
assignment should combine formula constraints, signal count, integration,
multiplicity, coupling constants, symmetry, and cross-nucleus consistency.

## 2. Scope and Data Assumptions

### 2.1 Supported inputs

The current project uses only:

- one 1D `1H NMR` spectrum image;
- one 1D broadband-decoupled `13C NMR` spectrum image;
- normalized 1H shifts, multiplicities, coupling constants, and integrations;
- normalized 13C shifts;
- optional molecular formula;
- solvent and spectrometer frequency as metadata.

The molecular formula is an explicit input field. During training it may be
derived from the target SMILES while the dataset is built. During inference it
must come from the user. If it is missing, the formula and DBE rules are
skipped. The prompt and rule engine must never reconstruct it from a hidden
reference SMILES.

### 2.2 Explicit exclusions

This project does not use DEPT, APT, COSY, TOCSY, HSQC, HMBC, NOE/NOESY, or
other multidimensional or edited experiments. The rule engine must not infer
correlations that are unavailable in the input.

The generated images do not contain simulated solvent, reference, or impurity
peaks. Therefore:

- no solvent-peak removal rule is applied;
- the absence of a solvent peak is not evidence of sample purity;
- solvent metadata may describe the source measurement but is not itself a
  structural signal.

The rendered spectra originate from normalized peak tables. Conclusions from
this benchmark do not automatically transfer to raw instrument spectra with
phase errors, baseline drift, spinning sidebands, truncation, impurities, or
unresolved solvent peaks.

### 2.3 Element policy

Dataset molecules are restricted to:

```text
H C N O F Si P S Cl Br I
```

Any molecule containing another element is excluded before subset curation.
The exclusion report and manifest retain the rejected element symbols for
auditability.

## 3. Evidence Strength and Interpretation Policy

Each machine rule has a stable ID, category, conclusion, confidence, strength,
human interpretation note, and structured metadata. Four strength levels are
used:

| Strength | Intended meaning | Examples |
|---|---|---|
| `hard` | Deterministic under the stated assumptions | formula composition, DBE |
| `strong` | Diagnostic combination with limited common alternatives | matched ethyl triplet/quartet J values |
| `moderate` | Useful when supported by independent evidence | heteroatom-adjacent shift regions |
| `weak` | Broad hypothesis-generating evidence | saturated alkyl shift region |

A hard rule may reject a candidate. A moderate or weak chemical-shift rule
should not reject a structure by itself. When rule context is added to a VLM
prompt, evidence is ordered `hard -> strong -> moderate -> weak` and truncated
to the configured maximum.

## 4. Recommended Human Workflow

Use the following order. Starting with an attractive substructure before
checking the global constraints is a common source of confirmation bias.

### Step 1: Check input integrity

1. Confirm that both nuclei have at least one valid signal.
2. Verify that every 1H peak has a plausible shift and integration.
3. Treat missing J values as unavailable, not as zero coupling.
4. Compare peak tables with rendered images for truncation or ordering errors.
5. Record broad, overlapping, or uncertain signals before assigning fragments.

### Step 2: Establish global formula constraints

1. Parse the element counts.
2. Calculate DBE when the formula is compatible with the common-valence model.
3. Note the total C and H counts.
4. Reserve heteroatoms and halogens for later fragment accounting.

### Step 3: Inspect the 13C spectrum first

1. Count distinct carbon environments.
2. Identify carbonyl, unsaturated, and heteroatom-substituted regions.
3. Compare observed signal count with formula carbon count.
4. Consider symmetry, accidental overlap, and weak quaternary signals.

The 13C spectrum gives an efficient global inventory. It often prevents an
incorrect local 1H interpretation from propagating into a full structure.

### Step 4: Partition the 1H spectrum into fragments

1. Normalize and compare integrations.
2. Match signals that share the same J value.
3. Identify reliable fragment patterns.
4. Keep ambiguous multiplets and exchangeable protons separate.
5. Avoid forcing every signal into an `n+1` pattern.

### Step 5: Assemble and falsify candidate structures

For each candidate, verify:

- exact element counts;
- complete DBE accounting;
- enough carbon atoms to support all observed 13C signals;
- agreement with strong fragment evidence;
- plausible symmetry and number of environments;
- an explanation for every diagnostic proton and carbon signal.

Actively search for contradictions. A candidate that explains one conspicuous
peak but leaves a carbonyl, aromatic region, or integration unexplained should
not survive.

## 5. Molecular Formula and DBE

### 5.1 Implemented expression

For the allowed elements and conventional common valences:

```text
DBE = (2C + 2Si + 2 + N + P - H - F - Cl - Br - I) / 2
```

Equivalent contribution form:

| Element | Common valence used | Contribution to `2 * DBE` |
|---|---:|---:|
| C, Si | 4 | +2 per atom |
| N, P | 3 | +1 per atom |
| O, divalent S | 2 | 0 |
| H, F, Cl, Br, I | 1 | -1 per atom |
| Constant | - | +2 |

Examples:

| Formula | DBE | Interpretation |
|---|---:|---|
| C2H6O | 0 | no required ring or pi bond |
| C6H6 | 4 | four total rings/pi-bond units |
| C8H8O | 5 | five total unsaturation units |
| Si2H6 | 0 | saturated common-valence silicon formula |
| C2H7P | 0 | saturated trivalent-phosphorus formula |

### 5.2 Limitations

Formula-only DBE can be ambiguous for unusual hypervalent P or S chemistry,
radicals, salts, disconnected formulations, and charged species. The current
parser accepts one neutral molecular formula. It rejects disconnected or
charged formula strings and refuses negative or non-half-integral DBE results.
It does not silently clamp them to zero.

### 5.3 Practical use

- DBE 0 rules out structures that require a ring, carbonyl, alkene, alkyne, or
  aromatic ring under the stated formula model.
- One ring or double bond consumes one DBE.
- One triple bond consumes two DBE.
- A benzene ring normally consumes four DBE, but DBE 4 does not prove that a
  benzene ring is present.
- The final candidate must account for all DBE units. Unassigned DBE is a
  strong indication that the structure is incomplete or wrong.

## 6. 1H NMR Chemical-Shift Evidence

The following ranges match the current `nmr_1d` rule library. They are deliberately broad and may
overlap.

| Rule ID | Range, ppm | Evidence | Important alternatives |
|---|---:|---|---|
| `H1_SHIFT_ALKYL` | 0.5-2.0 | saturated alkyl protons | many ordinary alkyl environments |
| `H1_SHIFT_PI_OR_CARBONYL_ADJACENT` | 1.7-3.3 | allylic, benzylic, propargylic, or carbonyl-alpha protons | several classes overlap strongly |
| `H1_SHIFT_HETEROATOM_SP3` | 3.0-4.6 | proton on an sp3 carbon near O, N, or halogen | acetal and other polarized environments |
| `H1_SHIFT_ALKENE` | 4.5-6.8 | vinylic proton | some acetal and heterocyclic protons |
| `H1_SHIFT_AROMATIC` | 6.0-9.0 | aromatic or strongly conjugated proton | heteroaromatic and highly conjugated systems |
| `H1_SHIFT_ALDEHYDE` | 9.0-10.6 | aldehydic proton | uncommon strongly deshielded environments |
| `H1_SHIFT_ACIDIC_OH` | 10.0-13.5 | acidic or strongly hydrogen-bonded proton | exchange-sensitive; concentration dependent |

### 6.1 How to use shift evidence

1. Use a shift to propose an environment, not to assign a unique group.
2. Combine it with integration and multiplicity.
3. Check whether the corresponding 13C region exists.
4. Check whether the formula contains the required heteroatom or DBE.
5. Retain overlapping alternatives until an independent observation resolves
   them.

## 7. 1H Integration

Integration estimates relative proton populations, but it is not universally
exact.

### Reliable uses

- distinguish approximate 1H, 2H, 3H, 6H, or 9H populations;
- test fragment ratios such as 3:2 for an ethyl group;
- compare aromatic proton count with the proposed substitution pattern;
- scale the total integral against formula hydrogen count when all protons are
  expected to be observed.

### Common failure modes

- overlapping environments merge their integrals;
- OH and NH signals may exchange, broaden, move, or disappear;
- incomplete relaxation and processing can distort areas;
- a reported integer may already be rounded or normalized;
- equivalent groups can combine into one large integral;
- diastereotopic CH2 protons may appear separately rather than as one 2H
  signal.

Treat small integration discrepancies as uncertainty. Treat a large global
hydrogen-count contradiction as evidence against the candidate.

## 8. Multiplicity and Coupling Constants

### 8.1 Limits of the `n+1` rule

The `n+1` rule assumes a first-order system with `n` equivalent neighboring
spin-1/2 nuclei and similar coupling relationships. It may fail for:

- strong or second-order coupling;
- chemically equivalent but magnetically non-equivalent protons;
- multiple distinct J values;
- diastereotopic methylene protons;
- unresolved overlap;
- exchangeable protons;
- simplified source annotations such as `m` or `br s`.

Peak labels should therefore be interpreted together with the J values. A
triplet and quartet with unrelated J values do not establish an ethyl group.

### 8.2 Approximate coupling heuristics

These empirical ranges are useful to humans but are not encoded as hard
candidate-rejection rules:

| Coupling relationship | Typical magnitude, Hz | Use |
|---|---:|---|
| vicinal saturated H-C-C-H | about 5-10 | fragment connectivity and conformation |
| trans alkene | about 12-18 | supports trans relationship |
| cis alkene | about 6-12 | supports cis relationship |
| geminal alkene | about 0-3 | supports terminal or geminal relationship |
| aromatic ortho | about 6-9 | adjacent aromatic protons |
| aromatic meta | about 1-3 | longer-range aromatic coupling |
| aromatic para | often 0-1 | weak and frequently unresolved |

Ranges depend on geometry, substitution, ring strain, and measurement quality.
They should rank hypotheses, not establish stereochemistry alone.

## 9. Implemented 1H Fragment Rules

### 9.1 Ethyl-like fragment: `H1_FRAGMENT_ETHYL_001`

Required observations:

- approximately 3H triplet;
- approximately 2H quartet;
- at least one J value matching within 0.5 Hz.

Inference: supports `CH3-CH2-`.

Human check: inspect both shifts to determine what is attached to CH2. A CH2
near 3-4.6 ppm suggests polarization by a heteroatom; a benzylic or
carbonyl-alpha CH2 is usually further downfield than an ordinary alkyl CH2.

### 9.2 Isopropyl-like fragment: `H1_FRAGMENT_ISOPROPYL_001`

Required observations:

- approximately 6H doublet;
- approximately 1H septet, heptet, or compatible multiplet;
- matching J within 0.7 Hz.

Inference: supports `CH(CH3)2`.

Human check: the methine may be annotated as `m` when additional coupling or
overlap obscures a clean septet. Confirm the 6:1 integration relationship.

### 9.3 tert-Butyl-like fragment: `H1_FRAGMENT_TERT_BUTYL_001`

Required observation: approximately 9H singlet.

Inference: supports three equivalent methyl groups attached to a center with
no directly coupled proton.

Human check: the 13C signal count should be compatible with equivalent methyl
carbons and one central carbon. A 9H singlet is strong evidence, but it does
not identify the atom attached to the central carbon.

### 9.4 Methoxy-like fragment: `H1_FRAGMENT_METHOXY_001`

Required observations:

- approximately 3H singlet;
- shift between 3.0 and 4.3 ppm.

Inference: supports an isolated heteroatom-bound methyl group.

Alternatives include N-methyl, S-methyl, and other polarized isolated methyl
groups. The formula and 13C shift are required to refine the assignment.

### 9.5 Acyl-methyl-like fragment: `H1_FRAGMENT_ACYL_METHYL_001`

Required observations:

- approximately 3H singlet;
- shift between 1.8 and 2.8 ppm.

Inference: supports a methyl group adjacent to a carbonyl or conjugated center.

Human check: benzylic and other pi-adjacent methyl groups overlap this range.
Do not call an acetyl group unless the 13C spectrum supports a carbonyl.

## 10. 13C NMR Chemical-Shift Evidence

| Rule ID | Range, ppm | Evidence | Important alternatives |
|---|---:|---|---|
| `C13_SHIFT_ALKYL` | 0-50 | saturated alkyl carbon | broad, weakly diagnostic region |
| `C13_SHIFT_HETEROATOM_SP3` | 45-90 | heteroatom-substituted sp3 carbon | overlaps sp carbon region |
| `C13_SHIFT_ALKYNE` | 65-100 | sp-hybridized carbon | overlaps oxygenated and other polarized sp3 carbon |
| `C13_SHIFT_UNSATURATED` | 100-165 | alkene, aromatic, or conjugated sp2 carbon | multiple unsaturated classes |
| `C13_SHIFT_HETERO_CARBONYL` | 160-185 | acid, ester, amide, or related heteroatom carbonyl | conjugated heteroatom-bearing sp2 carbons may overlap |
| `C13_SHIFT_ALDEHYDE_KETONE` | 185-220 | aldehyde or ketone carbonyl | aldehyde assignment requires 1H support |

### 10.1 Signal count

For a valid formula-conditioned sample:

```text
number of distinct observed 13C signals <= number of carbon atoms
```

If the observed count is smaller, consider:

- molecular symmetry;
- equivalent repeated groups;
- accidental overlap;
- weak or missing quaternary carbon signals;
- incomplete source reporting.

If the observed count exceeds formula carbon count, investigate duplicate
entries, malformed ranges, inconsistent source records, or an incorrect
formula.

### 10.2 Intensity warning

Broadband-decoupled 13C peak intensity is generally not quantitative because
of relaxation and nuclear Overhauser enhancement differences. A weak peak is
not automatically a minor component, and a missing quaternary signal is
possible. The current rule library uses positions and counts, not 13C
intensity, to validate structures.

## 11. Symmetry, Equivalence, and Overlap

Symmetry is a global constraint and should be evaluated before deciding that
signals are missing.

- Fewer 13C signals than carbon atoms can indicate equivalent carbons.
- A large 1H integral can represent several equivalent groups.
- Para-substituted or otherwise symmetric aromatic systems may show fewer
  environments than an unsymmetrical analogue.
- Accidental overlap can mimic symmetry, especially in crowded alkyl or
  aromatic regions.
- Conformational averaging may make environments equivalent on the NMR time
  scale.
- Chirality or restricted rotation may make apparently similar protons
  non-equivalent.

Do not infer symmetry from a single overlap. Require a pattern of reduced 1H
and 13C environment counts that is compatible with the full candidate.

## 12. Cross-Spectrum Consistency

The strongest conclusions combine independent observations.

| Proposed feature | Expected supporting evidence |
|---|---|
| aldehyde | 1H near 9-10.6 ppm and 13C near 185-220 ppm |
| ketone | 13C near 185-220 ppm, without requiring aldehydic 1H |
| acid/ester/amide carbonyl | 13C near 160-185 ppm plus formula/1H context |
| alkene | 13C in 100-165 ppm and/or vinylic 1H in 4.5-6.8 ppm |
| aromatic ring | multiple unsaturated 13C signals and commonly 1H in 6-9 ppm |
| alcohol/ether/amine environment | polarized 1H in 3-4.6 ppm and/or 13C in 45-90 ppm |
| alkyne | DBE support and 13C in 65-100 ppm |
| nitrile | N in formula, DBE support, and 13C approximately 105-130 ppm |

The functional-group spectral-consistency metric applies only to groups with a
defined 1D signature. It reports the fraction of checkable predicted groups
that have compatible spectral evidence. It is deliberately soft and does not
replace exact structure metrics.

## 13. Candidate Structure Validation

The implemented candidate validator uses objective or strongly structured
checks:

1. the candidate must be a valid SMILES;
2. if formula is supplied, its element counts must match exactly;
3. candidate DBE must match formula DBE;
4. candidate carbon count must be at least the number of observed 13C signals;
5. diagnostic fragment patterns detected from 1H data should occur in the
   candidate structure.

Broad shift-region hypotheses are not used as automatic rejection criteria.
A candidate may receive a low soft-consistency score while remaining a valid
structure prediction.

For candidate-ranking supervision, negatives are required to share the target
molecular formula. They are selected by high Morgan similarity so that the
task tests spectral discrimination rather than trivial formula rejection.

## 14. Common Failure Patterns

### 14.1 Overusing one diagnostic peak

Failure: assigning an entire structure from one carbonyl or one downfield
proton.

Correction: require complete atom, DBE, integration, and signal-count
accounting.

### 14.2 Treating `n+1` as universal

Failure: interpreting every multiplet as first-order coupling.

Correction: use matching J values, inspect second-order risk, and preserve
ambiguous `m` annotations.

### 14.3 Ignoring equivalent environments

Failure: rejecting a symmetric molecule because it has fewer signals than
atoms.

Correction: compare both nuclei and test the proposed symmetry globally.

### 14.4 Treating 13C intensity as atom count

Failure: using peak height to assign CH3, CH2, CH, or quaternary carbon.

Correction: the current broadband-decoupled input does not support this
assignment. Use only shifts and environment counts.

### 14.5 Forcing exchangeable protons into fragments

Failure: using OH/NH multiplicity or integration as a rigid connectivity rule.

Correction: mark exchangeable protons as condition-sensitive and prioritize
non-exchangeable coupling evidence.

### 14.6 Confusing benchmark cleanliness with experimental cleanliness

Failure: concluding that no impurities or solvent are present because their
peaks are absent from the generated image.

Correction: the rendering pipeline omits them by design.

## 15. Worked Example: Ethanol-Like Pattern

Input summary:

- formula C2H6O;
- 1H: about 1.18 ppm, triplet, 3H, J about 7 Hz;
- 1H: about 3.65 ppm, quartet, 2H, J about 7 Hz;
- broad exchangeable 1H signal;
- 13C: about 18 and 58 ppm.

Reasoning:

1. DBE is 0, so no ring or pi bond is required.
2. The matched triplet/quartet J values and 3:2 integration support an ethyl
   fragment.
3. The 2H signal and the 58 ppm carbon support a heteroatom-adjacent CH2.
4. The formula contains one oxygen and no other heteroatom.
5. The remaining proton is compatible with exchangeable OH.
6. `CCO` satisfies the formula, signal count, fragment, and shift evidence.

The decisive observation is not the chemical shift alone. It is the joint
agreement of DBE, matched J values, integration, 13C count, and oxygen content.

## 16. Worked Example: Aromatic Ketone-Like Pattern

Hypothetical summary:

- formula C8H8O;
- DBE 5;
- several 1H signals in the aromatic region totaling about 5H;
- one approximately 3H singlet near the pi/carbonyl-adjacent region;
- multiple 13C signals in the unsaturated region;
- one 13C signal near 195-205 ppm.

Reasoning:

1. An aromatic ring accounts for four DBE units.
2. A carbonyl accounts for the fifth DBE unit.
3. The high-shift carbonyl carbon supports an aldehyde or ketone.
4. Absence of an aldehydic 1H signal favors a ketone over an aldehyde.
5. A 3H singlet adjacent to the carbonyl supports an acyl methyl.
6. Candidate structures must still reproduce the aromatic substitution pattern
   and the observed number of carbon environments.

This example illustrates why DBE and cross-nucleus evidence should be applied
before selecting a familiar fragment.

## 17. Machine-Readable Implementation Map

| File | Responsibility |
|---|---|
| `rules/nmr_1d.yaml` | current shift and fragment rules |
| `src/nmr_rules/formula.py` | formula parsing and DBE |
| `src/nmr_rules/models.py` | typed evidence records |
| `src/nmr_rules/engine.py` | evidence generation from 1D peak tables |
| `src/nmr_rules/validator.py` | candidate SMILES consistency checks |
| `src/data/functional_groups.py` | functional-group SMARTS ontology |
| `src/data/spectral_regions.py` | deterministic region labels |
| `src/data/tasks.py` | structure and auxiliary task prompts/targets |
| `src/evaluation/spectral_consistency.py` | soft group-to-spectrum checks |
| `src/evaluation/metrics.py` | structure, consistency, and behavior metrics |

## 18. Versioning and Maintenance

Any scientific rule change should:

1. receive a new stable rule ID;
2. state required inputs and exclusions;
3. distinguish hard constraints from soft evidence;
4. include a human-readable caveat;
5. add deterministic unit tests;
6. preserve old experiment configs for comparison;
7. be evaluated through an ablation rather than assumed beneficial.

Chemical-shift ranges should eventually be calibrated against distributions in
the filtered project dataset. The current values are initial expert ranges,
not learned probability densities.

## 19. Reporting Guidance

In a paper or technical report:

- describe the component as an auditable 1D NMR prior or structured spectral
  constraint library;
- report the exact rule library and functional-group ontology used;
- separate baseline, rule-context, formula-free, and multitask experiments;
- report Exact Match, valid SMILES rate, molecular-formula accuracy, Tanimoto,
  scaffold match with coverage, functional-group F1, and spectral consistency;
- report output-format compliance, illegal-structure rate, and non-SMILES rate;
- state that the benchmark images omit solvent and impurity peaks;
- do not claim complete manual structure elucidation capability from 1D data
  alone.

## 20. Reference Baseline

- IUPAC, *NMR nomenclature. Nuclear spin properties and conventions for
  chemical shifts*, https://doi.org/10.1351/pac200173111795
- NIST Chemistry WebBook, https://webbook.nist.gov/chemistry/
- AIST Spectral Database for Organic Compounds (SDBS),
  https://sdbs.db.aist.go.jp/

These sources provide nomenclature and reference spectra. The numerical ranges
in this rulebook remain soft project rules and should be validated empirically
against the final filtered dataset.
