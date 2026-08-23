# Observatory design QA

Status: passed — no open P0/P1/P2 findings.

- Source: `docs/design/reference-editorial-observatory.png` (1489×1058).
- Matched implementation: `docs/design/implementation-reference-size-v5.png` (1489×1058).
- Combined comparison: `docs/design/qa-comparison-v5.png`.
- Responsive captures: `implementation-1440-v5.png`, `implementation-1024-v5.png`, `implementation-768-v5.png`, `implementation-390-v5.png`.
- Density: desktop comfortable; compact rail at 1024; mobile navigation and vertically arranged graph at 390.
- Data state: one idempotent demo ingestion, one evidence-bound CAUSE answer, one exact source span.
- Interaction checks: query submit, tick playback, direct tick selection, active edge/node state, inspector selection, parameters drawer.
- Runtime checks: no horizontal overflow at 1440/1024/768/390; no browser console errors; backend trace complete.

Fix history:

1. Replaced the misleading `S/C/P/H/N` peer legend with the formal `AH=<S,C,P,H,L>` tuple and separated `N` as a hypernode structure.
2. Replaced opaque `CAUSE` cells with a readable proposition and visible role bindings.
3. Made tick playback update the active phase, edges, nodes, and excitation values.
4. Removed duplicated ingestion and working-memory label concatenation from answers.
5. Matched the supplied editorial observatory reference: direct query, plain answer, dominant graph, evidence inspector, attached tick timeline, and restrained surfaces.
