# Project 04 integration handoff

Use the RC1 saved Version 2 educational adapter and frozen native runtime.
Adapter SHA-256: `0ea705ee3eef93b7a93d6d0647408fdd75be04f4a9359063835151a02b38365b`.
Protocol: `ios-risk-native-release-candidate.1`. The model's original strict
qualification decision is BENCHMARK_GATES_FAILED. Project completion does not
change this model qualification result.

The native interface is `NativeRiskPredictor.load(adapter_path)` followed by
`predictor.assess_risk(facts, relations=...)` or `predictor.generate(row)`.
The result retains raw generation, prompt token IDs, protocol and validation.
Facts must be string-valued. Do not repair invalid responses invisibly.

For Project 04, wrap this interface in a typed outcome with model and artifact
hashes, request identity, supplied facts, raw answer, validation, policy decision,
review state and execution status. Preserve separate states for a model proposal,
a permitted action and an independently verified outcome. No financial action
should run because the model wrote `restrict` or `escalate`.

Use existing failures as regression fixtures: NaN counts described as proven
inequality violations, unknown intent changed to absent intent, reversed event
order, inventory and proceeds recipients confused, and invented legal rules.
Exact copied evidence and valid JSON do not establish correct rationale.
Unknown or unsupported claims should remain unresolved for review. Legal content
requires authoritative source verification; a schema check is insufficient.

The CPU replay uses saved outputs and is suitable for a tutorial without a GPU.
Live inference requires compatible CUDA dependencies and base-model access.
Historical evaluation outputs are exposed development fixtures for Project 04,
not fresh test evidence for future model improvement.
