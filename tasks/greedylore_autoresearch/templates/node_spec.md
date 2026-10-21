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

## Frozen research intent (provided via --intent-file)
- `problem_ref`: parent `Nxxx:Q1`, `Nxxx:Q2`, or `Nxxx:current`; null is allowed for EXPLORE or legacy evidence.
- `question`: the specific research question, with a substantive mechanism delta.
- `mechanism_test`: comparator, predicted observable and falsification condition.
- `practical_test`: separate quality and complete-cycle efficiency criteria.
- `handover_response`: why continue, challenge, defer or choose another question.
- `engineering_review`: relevant Engineering Summary/Analysis, applicability and reuse/defer decisions.
- The host freezes available parent handover content and report references; do not edit generated Specs.
