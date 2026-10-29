---
name: coral-greedylore-prompt-writing
description: Use when writing, revising, reviewing, or translating prompts for CORAL GreedyLoRE AutoResearch.
---

# CORAL GreedyLoRE AutoResearch Prompt Writing

Use this skill to write and maintain role, analysis, knowledge-curation, shared-rule, and startup prompts. Help agents understand the problem, available conditions, and expected deliverables, so they can exercise research judgment within those conditions. Optimize for clarity and executability rather than minimum length.

## 1. State the purpose and expected outcome first

Introduce each role through its background, responsibility, problem, and deliverable. Open larger sections with what the phase should accomplish, then explain available capabilities and grounds for judgment.

- Do not substitute abstract instructions for concrete purposes.
- Do not treat output field names as role objectives.

## 2. Map inputs and references

Explain the purposes of code, research records, evidence, and reference material, and identify authoritative sources. Important conclusions should trace to original evidence; operational instructions should match actual interfaces, permissions, and the current contract.

- Consult research references as needed rather than turning a resource catalog into a mandatory research route.
- Identify source versions, scope, and accessibility; do not require unavailable or inaccessible material.

## 3. Preserve research preferences and their strength

Explicitly retain user preferences concerning problem selection, inheritance, exploration routes, and research content. Use wording such as "prioritize," "favor," and "may consider" that matches the original intent.

The following content must currently be retained:

- Parent selection favors low-rank and sparsification directions.
- Research favors new methods and mechanisms.
- Retain supplementary guidance for combination/adaptive methods: combination methods explain their motivation and complementarity; adaptive methods explain the basis for adaptation, resulting behavior, and relevant state and costs.

Do not turn preferences into prohibitions or weaken them into generic freedom of choice. Assign preferences according to role responsibilities while preserving independent analytical judgment.

Distinguish encouragement, permission and prohibition in both wording and emphasis. When direct copying is forbidden and borrowing ideas is allowed, lead with the prohibition and express borrowing as permission rather than advice to use existing work.

## 4. Give autonomy executable capability boundaries

Explain what agents can do, why they would do it, and under which conditions. Preserve real phase dependencies while leaving methods, investigation order, and depth of effort to the agent.

Judge instructions by the information they add to the supplied role, task and capabilities. Omit explanations already clear from those inputs, including generic reminders to act autonomously; retain explicit permissions and non-obvious boundaries needed for independent execution.

- Do not reduce autonomy to a fixed sequence.
- Do not omit necessary conditions when removing micromanagement.
- Do not turn optional capabilities into mandatory activities or additional promotion gates.

## 5. Provide real commands for required operations

Give actual commands for required operations, with references for complex fields and complete examples. Distinguish recording a judgment, executing an action, and completing publication.

- Do not invent commands, fields, permissions, or recovery capabilities.

## 6. Constrain research quality through purpose

Require enough evidence to answer the research question, distinguish consequential explanations, and assess practical quality and cost. Let agents choose log analysis, derivations, checks, controlled experiments, and their order.

Connect requested research activities to the uncertainties they address and judgments they inform. Mechanism work, for example, should help explain observations, distinguish competing explanations and guide method refinement or future hypotheses.

- Preserve original predictions, reasons for revision, failures, counterevidence, unexpected gains, and important unknowns.
- Distinguish mechanism explanations from practical value; do not make mechanism confirmation an additional promotion gate.

## 7. Use prose for requirements and bullets for constraints

Use prose to explain tasks, purposes, operations, and deliverables; group longer content under subheadings. Use bullets for conditions, prohibitions, hard limits, and easily misunderstood boundaries, adjacent to the relevant operation. Tables suit resource maps, capabilities, and output fields.

- Avoid dense blocks, sentence-by-sentence bulleting, and empty sections.
- Translation preserves heading levels, paragraph groups, prose/bullet divisions, and wording strength; do not restructure while translating.

## 8. Assign rules by responsibility

Startup entry points cover identity, location, current state, and role entry points; role documents cover objectives, preferences, capabilities, and deliverables; shared rules cover cross-role protocols; command references cover interface details; the host handles counting, scheduling, state, and publication.

- Avoid injecting complete duplicate instructions through multiple entry points.
- Independent sessions need their own necessary boundaries and output contracts.
- After deduplication, every requirement must retain an explicitly reachable location.

## 9. Preserve system interfaces and non-obvious constraints

Clearly express existing execution protocols, permissions, and non-obvious constraints. Explain distinctions between easily confused states, execution behavior, and recovery semantics where needed.

Preserve each constraint's scope, conditions, exceptions and force when shortening it. Keep behavior-changing details such as diagnostic isolation, fresh-data confirmation, timing reuse and result selection.

- Do not change budgets, eligibility, permissions, schemas, or evaluation protocols through editorial revision.
- Interpret historical runs under their saved contracts.
- Use 1800-second blocking waits for long-running tasks, allowing early return and continuation when the task ends. Preserve the original session and request identity to avoid duplicate execution and meaningless polling. Subsequent revisions must not omit or generalize away the explicit 1800-second requirement.

## 10. Make knowledge outputs show information gain and scope

Retain findings, counterexamples, and reliable effects that can influence design, experiments, or practical judgment. Merge evidence for the same insight; narrow, revise, or retract earlier conclusions when evidence conflicts.

- Do not require new findings or knowledge updates every time.
- Repeated observations do not automatically broaden conclusions, and inherited coverage does not mean fresh inspection.
- Do not substitute run status or generic limitations for research findings.

## 11. Review revisions with a coverage checklist

Check the original task requirements, hard constraints, research preferences, optional capabilities, required commands, and output contracts. Identify their locations in the revised text or other reachable inputs.

For local revisions, prefer the smallest edit that resolves the issue: add a missing purpose, remove redundant text or shorten the affected constraint. Preserve accurate surrounding prose, structure and research preferences.

- Check omissions, weakening, unintended strengthening, scope changes, inaccessible references, and contradictions.
- First verify that the original text agrees with the current contract.
- Treat length changes as supplementary information.
- Explicitly check the three research requirements in Section 3 and the 1800-second waiting requirement in Section 9, ensuring their complete preservation in revisions and translations.

## 12. Separate textual review from runtime effectiveness

Compare clarity, executability, semantic coverage, room for independent judgment, and maintenance cost. Claims about actual agent performance require separate runtime comparisons.

- Textual review does not establish improvements in research quality, experimental outcomes, or token usage.
- Retain traceable revision and review evidence; when replacing actual entry points, verify delivered content, references, and relevant existing checks.
