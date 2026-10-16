node_id: NXXX
node_type: research         # research | seed；仅预先给定的 Root 使用 seed
parent_node: NYYY           # seed Root: null
base_commit: "<full_git_sha>"
parent_research_analysis: "NYYY/research_analysis_v1.md"   # seed Root: null
parent_engineering_analysis: null                          # 可以晚到
---
# Node: <short name>

## Existing baseline / observed gap
- `seed` Root：写明提供的 baseline 算法、源代码基准 SHA、benchmark/config/来源版本，不填写虚构的 parent gap。
- 普通 Node：引用 parent 证据。

## Parent selection rationale (仅普通 Node，1–3 句)
- <为什么选择该 Node / base commit：预期收益、证据、探索程度、成本/风险；引用实际 Analysis 版本>

## Observed gap (仅普通 Node，引用 parent 证据)
- 证据：<parent analysis / attempt>
- 尚未解决的问题：...

## Hypothesis (普通 Node，需可证伪；seed Root 写「provided baseline」)
...

## Design delta (普通 Node：相对 base commit 的算法改动；seed Root：none)
...

## Expected quality / efficiency effect & risks
...

## Falsification criteria
...
