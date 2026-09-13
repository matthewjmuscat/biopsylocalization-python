# Working on this scientific platform

Before substantial changes, read [project north stars](docs/architecture/PROJECT_NORTH_STARS.md),
[scientific standards](docs/architecture/SCIENTIFIC_CODEBASE_STANDARDS.md), the
[process architecture](docs/architecture/PATIENT_RUNNER_PROCESS_ARCHITECTURE.md),
and relevant documents in [the documentation index](docs/DOCUMENTATION_INDEX.md).
Inspect current code and evidence; older conversation summaries are context.

- Build a trustworthy, auditable, modular scientific platform with composable
  pathways, not one permanently fixed pipeline. Preserve existing capabilities.
- Separate scientific algorithm changes from execution/config/interface changes.
  Preserve the legacy oracle until replacements are explicitly validated;
  historical behavior is not automatically scientifically desirable.
- Keep the parent lightweight and ordinary workers isolated and patient-scoped.
  Scientific modules own algorithms; interfaces request dependency-valid work.
- Distinguish permanent contracts, transitional dictionary/config adapters, and
  disposable legacy validation tools. Migration must converge: extract config
  from main, cut main orchestration, replace master dictionaries at validated
  stage boundaries, and expose existing modular scientific producers.
- Complete approved work coherently. Do not silently expand scope. Surface useful
  improvements, risks, tools, and research hypotheses as now/soon/later
  recommendations, distinguishing recommendations from approved implementation.
- Use focused synthetic/contract tests and proportionate user-operated numerical
  gates. Avoid repetitive patient campaigns and speculative frameworks.
- Preserve strict provenance, explicit failures, and fresh outputs. Do not relax
  tolerances or rewrite historical evidence to manufacture agreement.
- Do not inspect or execute real patient data without explicit authorization;
  routine development uses synthetic fixtures. Patient execution stays user-run.
- Explain plainly what starts a run, what parent and worker hold, where config
  comes from, what is written, and what consumes it.
- At substantial handoffs include a short, useful north-star review: permanent
  progress, remaining transitions, meaningful risks/opportunities, and recommended
  roadmap changes. Avoid boilerplate or invented research findings.
