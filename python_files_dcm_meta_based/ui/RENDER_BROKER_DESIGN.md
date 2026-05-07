# Render Broker Design

## Purpose

This module family defines a render-control broker that sits between computational code and interactive render backends.

The broker exists so that rendering can be:

- requested from anywhere in the programme through a stable orchestration contract,
- reviewed or suppressed by the user before windows or exports open,
- configured at the point of use without pushing GUI code into numerical modules,
- migrated later to a different GUI framework without rewriting the render-call sites.

The current implementation target is optimizer-v2 only.

That first adoption is intentional.

Optimizer-v2 already has replayable render-job payloads, explicit scene-group semantics, and a narrow live integration seam, so it is the lowest-risk place to prove the broker contract before migrating older ad hoc Open3D plotting paths.

## Core Design Rules

### 1. GUI agnostic core, GUI specific adapters

The broker is split into two layers.

1. `ui/render_broker.py`
   This file owns the stable request, option, export, timeout, and session-state contracts plus the generic render-review loop.

2. `ui/tk_render_broker.py`
   This file owns one concrete dialog adapter implemented with Tkinter.

The optimizer or any future render call site talks only to the broker contracts.

The computational layer must not depend on Tk widgets, Tk variables, modal-dialog details, or any other toolkit-specific types.

That keeps the seam open for a later migration to a more research-facing GUI stack such as Qt, DearPyGui, wxPython, or a web UI.

### 2. Render code remains separate from render control

The broker does not render anything itself.

It decides whether a render should happen and with what settings.

Actual rendering continues to live in the owning renderer module.

For optimizer-v2 that renderer remains:

- `biopsy_optimizer/v2/render.py`

This means the broker chooses:

- which option group to act on,
- which specific scenes or candidates are selected,
- which backend to use,
- whether export is enabled,
- which export settings override the defaults,
- whether execution continues immediately.

The renderer still owns:

- Open3D scene creation,
- Plotly figure construction,
- Plotly static export,
- layer and camera application.

### 3. Call sites own domain-specific choices

The broker is intentionally generic.

It does not know what a stage boundary is, what a candidate replay is, or what a dose lattice is.

Each caller builds its own choice groups and action handler.

For optimizer-v2, the live integration layer owns:

- the stage-boundary option list,
- the candidate-containment replay option list,
- the mapping from a selected option to a concrete render job,
- the mapping from export overrides to `OptimizerV2PlotlyExportConfig`.

That pattern is what should be reused later by other render sites.

## Broker Contract

The broker request contains one or more choice groups.

Each choice group defines:

- a stable group key,
- a display label,
- explanatory text,
- one of `single` or `multi` selection mode,
- the selectable options,
- which backends are allowed,
- whether Plotly export is allowed,
- default export resolution and file-format settings,
- the button label that triggers rendering for that group.

Each option defines:

- a stable option key used by the caller,
- a human-readable label,
- whether it should be preselected,
- an optional suggested export directory.

The dialog returns one of two actions.

1. `continue`
   No render is executed and the programme continues.

2. `render`
   The caller receives a group key, selected option keys, a backend choice, and optional export overrides.

The caller then performs the render and re-enters the broker loop.

## Timeout Model

The timeout contract is part of the broker request, not the renderer.

The intended unattended behavior is:

- if a timeout is configured and the user does nothing, the broker returns `continue`,
- no windows or exports are launched automatically on timeout,
- the user can request more time from the dialog,
- the user can disable the timeout for the rest of the run.

This is safer than defaulting to auto-render because unattended exports or windows are usually the more disruptive failure mode.

The timeout policy therefore exposes:

- initial timeout in seconds,
- extension increment in seconds,
- whether extension is allowed,
- whether timeout disablement is allowed for the rest of the run,
- timeout action, currently `continue`.

Session state tracks only the cross-dialog policy mutation that matters right now:

- whether timeout has been disabled for the rest of the run.

Session state is intentionally generic and can later accumulate:

- skip current patient,
- skip current ROI,
- skip all remaining debug renders,
- remembered backend or export defaults.

## Export Model

Export belongs in the broker contract because export is a user-facing decision, not just a static config toggle.

The export design separates:

1. default export suggestions owned by the call site,
2. export overrides chosen live in the dialog,
3. render-backend execution owned by the renderer.

For optimizer-v2, the call site supplies suggested default directories based on:

- scene group,
- patient UID,
- ROI name,
- run output directory.

The dialog then allows the user to override:

- output directory,
- file types,
- width,
- height,
- scale.

The renderer receives the resolved export config and continues to own file emission.

This keeps the export UI flexible while avoiding duplicate file-writing code.

## Optimizer-v2 First Adoption Scope

This pass migrates only the optimizer-v2 live render path.

That includes:

- stage-boundary replay scenes,
- winner and non-winner candidate-containment replay scenes,
- backend choice inside the dialog,
- export choice inside the dialog,
- timeout inside the dialog,
- re-open-until-continue loop.

This pass does not migrate the older direct Open3D debug calls elsewhere in the repository.

Those will be moved later once the broker contract has been proven stable in optimizer-v2.

## Migration Strategy For The Rest Of The Codebase

Later migrations should follow the same pattern.

1. Identify a render call site.
2. Extract its render payload into replayable render jobs or a comparable stable request object.
3. Build broker choice groups at the orchestration seam.
4. Route user selection through the shared broker.
5. Keep the computational and rendering math unchanged underneath.

This is the path that eventually supports a full GUI application.

The current Tk dialog is only one adapter.

If a more professional GUI framework is adopted later, the expected migration should mostly be:

- replace the adapter implementation,
- preserve the broker contracts,
- preserve the call-site request builders,
- preserve the renderer modules.

That is the main reason the broker core must stay toolkit-agnostic.

## Practical Implementation Notes

- The broker should never import optimizer-v2 modules.
- The broker should never import Open3D or Plotly.
- Numerical modules should never import Tkinter.
- Export format validation should remain close to the renderer so unsupported file types fail where the writer contract actually lives.
- Suggested export locations should be caller-owned because only the caller knows run-specific output semantics.
- Timeout should default to disabled unless the caller opts in with a concrete number of seconds.

## Current Status

After this pass:

- optimizer-v2 uses the shared broker contracts,
- Tkinter is only an adapter,
- export path and resolution overrides are user-configurable inside the dialog,
- timeout is supported and can be disabled for the rest of the run,
- the system is positioned for later migration of legacy render paths without redesigning the render-control contract again.