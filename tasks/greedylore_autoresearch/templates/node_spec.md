node_id: NXXX
node_type: research         # research | seed; only the predefined Root uses seed
research_mode: REFINE       # host-assigned REFINE | EXPLORE; never an Agent request field
parent_node: NYYY           # seed Root: null
base_commit: "<full_git_sha>"
parent_research_analysis: "NYYY/research_analysis_v1.md"   # seed Root: null
parent_engineering_analysis: null                          # may arrive later
---
# Node: <short name>

## Existing baseline / observed gap
- `seed` Root: record the provided baseline algorithm, source SHA, benchmark/config and source version; do not invent a parent gap.
- Ordinary Node: cite parent evidence.

## Parent selection rationale (ordinary Nodes only, 1–3 sentences)
- <Why this Node / base commit: expected benefit, evidence, exploration coverage and cost/risk; cite the actual Analysis version.>

## Observed gap (ordinary Nodes only; cite parent evidence)
- Evidence: <parent analysis / attempt>
- Unresolved issue: ...

## Hypothesis (falsifiable for ordinary Nodes; write "provided baseline" for seed Root)
...

## Design delta (ordinary Node: algorithm change relative to base commit; seed Root: none)
...

## Expected quality / efficiency effect & risks
...

## Falsification criteria
...

## Host-generated research origin and initial design
- Researcher registers once with `origin` and first `design`, using `templates/research_origin.json`; it does not write this Spec or a separate research-intent file.
- `origin` records the question/hypothesis, evidence and selection rationale, parent problem response, risks/falsification and Engineering review.
- The immutable first design records method, mechanism, predictions and tests. Host-generated compatibility criteria reference its exact path and hash; `research_origin_version: 3` identifies this protocol.
- Later designs and observations/decisions append to `research.json`; they do not rewrite Spec or original predictions. The host freezes available parent handover and report references.
