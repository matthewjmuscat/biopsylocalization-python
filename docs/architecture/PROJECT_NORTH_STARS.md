# Project North Stars

The destination is a trustworthy, auditable, patient-scoped scientific platform
supporting multiple biopsy-localization, uncertainty, classification,
optimization, dosimetry, imaging, guidance, and future research/translational
workflows. It is not optimized around one paper, uncertainty model, or historical
pipeline order. This direction does not authorize unrelated implementation work.

## Scientific purpose and stewardship

Scientific correctness comes first. Make assumptions and behavior explicit,
distinguish algorithm changes from execution changes, and expose hidden coupling.
The legacy pipeline is retained reference evidence, not an assertion that every
historical behavior is scientifically desirable.

A completed run should be explainable through patient/input identity, resolved
scientific config and state, source/environment identity, pathway, seeds, stage
outcomes, warnings, artifact inventory, and relevant numerical evidence. Another
researcher should be able to use and extend it without private chat history.

Use the [process architecture](PATIENT_RUNNER_PROCESS_ARCHITECTURE.md): resolved
config/inventory → lightweight parent → isolated patient worker → scientific
stages → durable artifacts → post-run assembly/comparison. Patient scientific
state belongs in workers. Truly cohort-dependent methods must declare cohort
inputs rather than acquire them accidentally.

CLI, Python, batch, future APIs, and Studio Superposition should consume these
same boundaries. A GUI must not own scientific logic or manipulate incidental
runtime dictionaries. Remote scheduling is a possible future consumer, not a
reason to build a distributed system prematurely.

## Capability and pathway composability

Existing science includes biopsy preprocessing, optimizer v1/v2,
sampling/classification, dose and MR mapping, MC simulation, guidance planning,
and translations/rotations/dilations and relative-structure/biopsy transforms.
Dilation already supports contour/segmentation uncertainty investigation. Future
deformation/correlation models extend one subsystem, not a privileged north star.

Dependencies must represent actual scientific requirements and producer outputs.
A guidance workflow should eventually require prepared geometry and appropriate
target/core products without unrelated MC dosimetry or cohort analysis. Other
workflows may request classification, dosimetric/DVH QA, or optimizer and
realization/sampling branches. These are use cases, not API names to freeze now.

Use the [dependency graph](PATIENT_RUNNER_DEPENDENCY_GRAPH.md) and
[module ownership guide](PATIENT_MODULE_TREE_GUIDE.md). Keep scientific modules
passive and reusable; orchestration chooses dependency-valid subsets. Separate
algorithms, config, discovery, runtime construction, scheduling, artifacts,
validation, assembly, and presentation.

## Migration must converge

Validation enables replacement and deletion. Identify permanent, transitional,
reference-only, and removable components. Avoid both speculative frameworks and
repetitive bespoke experiment systems. Keep these migration tracks visible:

1. **Configuration:** finish production defaults/config construction outside
   main. Typed config remains authoritative; export-only main is transitional.
   Never duplicate scientific defaults in a validation recipe.
2. **Main:** remove responsibilities as replacement services become usable and
   validated. Retain an explicit oracle route; moving a monolith is not enough.
3. **Runtime state:** replace master reference/info dictionaries and obsolete
   containers through typed stage inputs/outputs with numerical evidence. Avoid
   indefinite duplicate mutable stores and full-runtime serialization contracts.
4. **Scientific modules:** expose already modular producers through independent
   workers and actual dependency contracts. Adapter existence does not establish
   standalone scientific parity.
5. **Retirement:** delete bridges/sidecars individually when replacement numerical
   and output coverage satisfies the scientific standards.

Performance matters alongside correctness: use bounded memory, explicit CPU/GPU
ownership, and durable products; permit later concurrency without changing
semantics. Optimize measured bottlenecks. Adopt established or modern tools when
they materially improve correctness, maintainability, usability, interoperability,
or scientific capability, not for novelty.

## Proportionate development and active review

Complete the task, challenge assumptions when warranted, and recommend better
paths with their cost, scientific risk, and benefit. Do not silently implement
unrelated cleanup or new research. Distinguish implement now, leave room for now,
and record for later. Prefer structural guarantees and focused synthetic tests
when they reveal mechanisms more directly than repetitive real-patient campaigns.

At substantial handoffs report meaningful permanent progress, transitions,
simplification opportunities, risks/debt, useful tools, and roadmap changes.
Surface scientific hypotheses and new workflow opportunities, separating them
from validated findings. Skip empty checklist items; understandable progress and
development velocity matter as well as caution.

## Opportunities recorded during the independence phase

- **Soon: production config construction.** The new gate consumes snapshots;
  extracting defaults remains separate behavior-preserving work that can proceed
  while user-operated gates run.
- **Soon: guidance producer contracts.** `guidance_maps.planning` already has
  one-patient non-plotting planning. Its workflow requires prepared/finalized
  target cores, while the executable guidance DAG node lists only anatomical and
  biopsy preprocessing. Specify actual target/core producers before exposing a
  prospective pathway, and separate coarse grid prerequisites where justified.
  No graph or scientific dependency changes in this phase.
- **Soon: typed grid/geometry products.** Use retained evidence to choose a small
  typed boundary replacing dictionary reads; do not rewrite all state at once.
- **Later: uncertainty contracts.** Preserve existing transforms while cleaning
  their representation and supporting independently validated new models.
- **Research hypothesis:** effective-state differences may affect derived
  products or visualization. Characterize actual affected fields before claiming
  scientific impact. Equal thresholds cannot disprove legacy threshold carry.
