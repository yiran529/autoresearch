decision: pass   # pass | warning | fail；正式生成时只填一个值
---

# <Research | Engineering> Review — <node_id> / vN

**Mode:** research | engineering
**Analysis reviewed:** `nodes/<node_id>/<mode>_analysis_vN.md`
**Research SHA:** `<full SHA>`
**Engineering focus (optional):** `<full SHA or none>`
**Checked evidence:** <fixed Attempt SHA + eval log/test paths>

## Decision / rationale
- <简短说明为什么是 frontmatter 中的 decision；不能同时填写多个状态>

## Workflow checks
- <唯一 Research Attempt / quality+efficiency 的完成性，或工程效率迭代/semantic 自检的真实性>
- <commit、config、日志和 Node 关联是否一致>

## Analysis checks
- <观察和解释是否有相应证据；替代解释、局限和结构性 efficiency 归因是否合理>

## Issues / actions
- <severity, 证据, 谁负责修正何处；没有则写 none>

> Reviewer 只写 Review 文件，不改变 Analysis、代码或评测结果。Quality 未达标不是自动 fail；soft semantic pass 不是严格等价证明。
