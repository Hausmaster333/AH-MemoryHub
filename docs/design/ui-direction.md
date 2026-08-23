# AH-MemoryHub UI Direction

Status: selected visual direction for implementation.

Visual reference: `docs/design/reference-editorial-observatory.png`.

## Product feeling

AH-MemoryHub should feel like an expensive scientific instrument: calm, exact,
credible, and pleasant to use under pressure. Premium quality comes from
typography, proportion, alignment, evidence density, and restrained motion—not
from black-and-gold styling, glass, gradients, or AI theatrics.

The user should understand one promise within seconds: the system does not just
answer; it shows the exact path by which memory reached the answer.

## Primary experience

The main screen is **Memory Observatory**. Its visual hierarchy is:

1. question field and one clear run action;
2. evidence-bound answer;
3. selected causal subgraph and highlighted activation path;
4. exact source evidence and selected-element details;
5. tick timeline with playback and scrubbing;
6. parameters as a secondary drawer.

The graph is never the whole database. It is a selected, readable projection of
the current reasoning path. Active nodes and links are prominent; inactive
context recedes. A user can hover for a preview and click to lock inspection.

## Required product surfaces

Keep the application small and coherent. Design only these three product
surfaces plus their essential states:

- **Memory Observatory** — query, answer, causal subgraph, source evidence,
  selected element, ticks, playback, and parameter drawer.
- **Ingestion Review** — add a document or fact, inspect accepted/rejected
  Candidate Facts, source spans, templates, and Role Bindings, then admit valid
  facts through the normal API.
- **Evaluation** — M1–M5 results and one AH-vs-vanilla-RAG comparison, presented
  as evidence for judges rather than a generic KPI dashboard.

Memory statistics, export, GC preview/commit, DSL, and settings remain compact
secondary tools. Do not turn each into a separate dashboard.

## Visual system

- Canvas: warm alabaster `#F4F1EA`.
- Primary surface: `#FBFAF7`.
- Main ink: `#17191A`; secondary ink: `#656A67`.
- Hairline: `#D9D6CF`.
- Primary action and activation: deep cobalt `#1261D8`.
- Memory-type accents, used sparingly and always with a letter marker:
  `S #257F7A`, `C #C45B35`, `P #2367D1`, `H #7651A8`, `N #2F7E91`.
- Typography: Inter for interface and IBM Plex Mono for UID, source coordinates,
  and trace values. Body text 14–16 px. No more than two fonts.
- Spacing follows an 8 px base rhythm. Prefer alignment and whitespace before
  containers. Use hairline dividers before borders, and shadows only for a
  selected floating element.
- Corners are restrained: 6–10 px for controls, 12 px maximum for a selected
  node or drawer. Avoid pill-shaped UI except compact status chips.
- Icons come from one coherent open icon family; no emoji or custom glyph art.

## Desktop composition

At 1440 × 1024:

- navigation rail: 176–192 px;
- central workspace: fluid and visually dominant;
- evidence inspector: 352–384 px;
- query bar aligned to the central workspace;
- timeline attached to the bottom of the central workspace, not a floating card.

The selected source excerpt and UID/tick data must be readable at presentation
distance. Do not show more than roughly 10–18 graph nodes in the initial answer
state.

## Responsive behaviour

- `>= 1280 px`: full navigation, central canvas, persistent evidence inspector.
- `960–1279 px`: icon navigation; inspector becomes a 320 px overlay drawer;
  graph automatically fits the remaining viewport.
- `720–959 px`: top navigation; answer above the trace; evidence opens as a
  bottom sheet; graph may switch to a readable causal-path layout.
- `< 720 px`: supported for resilience, not the presentation target. Stack the
  answer, causal path, evidence, and timeline; never allow page-level horizontal
  overflow. Advanced parameters live in a full-height sheet.

Resizing must preserve the selected tick, query, answer, and selected UID.

## Motion and states

- Tick transition: 180–260 ms with restrained easing.
- Activation is communicated by border, fill, and line emphasis—not glow alone.
- Pause animation when the tab is hidden. Honour `prefers-reduced-motion`.
- Provide authored empty, loading, insufficient-evidence, validation-error,
  backend-unavailable, and success states.
- Loading uses stable skeleton geometry; avoid layout jumps and perpetual
  spinners.
- Keyboard focus is visible; colour is never the only type/status cue.

## Non-negotiable exclusions

No neon AI aesthetic, 3D graph, particle field, glassmorphism, card wall,
black-and-gold luxury, giant marketing headline, fake metrics, universal graph
editor, or dense hairball. Do not hide provenance, UID, trace completeness, or
engine profile to make the screen cleaner.

## Design acceptance

The `.pen` file must contain reusable tokens/components and named frames for the
three surfaces. It must include at least the main answer state, ingestion review,
evaluation, empty/loading/error states, and one compact responsive frame. Export
PNG previews, inspect them for clipping and spacing errors, and correct all
material defects before handing the design to frontend implementation.
