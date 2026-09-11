# RC1 and GPT-4.1 matched benchmark

The adapter demonstrates a classification advantage on this simulated benchmark.
It also reproduces the project's reference-answer contract more closely in risk
and taught-section recall. GPT-4.1 provides more consistently supported risk
reasoning and substantially better unseen-section citation identification.
These results do not support general superiority over GPT-4.1.

| Metric | RC1 adapter | GPT-4.1 |
|---|---:|---:|
| Classification F1, 200 cases | 89.66% | 63.80% |
| Classification precision | 88.35% | 82.54% |
| Classification recall | 91% | 52% |
| Risk tier accuracy, 100 cases | 96% | 99% |
| Risk pattern accuracy | 88% | 100% |
| Risk supported rationale, internal review | 66% | 99% |
| Risk exact reference-evidence coverage | 99% | 8% |
| Risk full-answer checklist accuracy | 61% | 8% |
| Risk unsupported-claim flags | 28% | 1% |
| Required abstention shape, 20 cases | 100% | 100% |
| Taught-section citation accuracy, 40 cases | 100% | 90% |
| Taught-section citation plus substance | 97.5% | 15% |
| Unseen-section citation accuracy, 40 cases | 0% | 80% |
| Unseen-section citation plus substance | 0% | 7.5% |
| Unseen-section materially false rule flags | 90% | 17.5% |

## What the scores mean

Classification is the clearest advantage. The adapter detects 91/100 source
fraud labels with 12 false positives; GPT detects 52/100 with 11 false positives.
Both emit valid labels for every case. A paired customer-cluster bootstrap gives
a nominal 98.75% interval of +14.75 to +37.84 percentage points for the F1
difference, with 20,000 draws and four-primary-comparison multiplicity adjustment.

Risk full-answer scoring requires correct tier, pattern, action, every exact
reference evidence field and supported reasoning. Both models have 100% valid
risk outputs. GPT satisfies reference-evidence completeness on only 8/100,
despite 99/100 supported rationales and 99/100 correct tiers. The prompt's minimum
evidence requirement is less demanding than complete reference-field coverage.
The adapter's 61% versus 8% therefore demonstrates closer adherence to this
reference contract, not superior general reasoning. The paired interval for the
checklist difference is +15.0 to +88.5 points across 17 scenario groups.

Regulatory scoring separately checks a single citation occurrence and a complete
explanation of requested source provisions. Extra/repeated citations fail the
strict format; omitted required qualifications fail substance even when the
remaining statements are true. The taught-section joint difference interval is
+55 to +100 points across eight sections. Unseen-section joint difference is
-30 to 0 points: no adapter advantage. Citation-only transfer favors GPT clearly.
Do not call taught-section recall broad legal competence.

## Method and preserved evidence

The fixed adapter is RC1 saved Version 2, SHA-256
`0ea705ee3eef93b7a93d6d0647408fdd75be04f4a9359063835151a02b38365b`.
The provider returned `gpt-4.1-2025-04-14`. All 380 original scored cases were
used; 110 development cases were excluded. Both models received the same task
information, without retrieval or tools; numeric generation caps were 64/384/512.
Provider-specific tokenizers and templates differ. The existing adapter was
trained for this contract; no GPT-specific examples or prompt optimization were
provided. This evaluates the declared zero-shot protocol, not each model's best
possible configuration. It is a retrospective comparison on already inspected
RC1 cases, not a new prospective confirmation study.

The API first produced 335 successes and 45 HTTP 429 failures; one disclosed
sequential retry recovered all 45. All 380 successful responses ended normally.
Internal AI reviewers saw shuffled mixed outputs with model identities and
authored targets hidden, using supplied facts and official dated source text.
All semantic judgments and severe failure causes are hash-bound. They are not
qualified independent professional validation.

Re-review changed some subjective flags on identical RC1 answers: supported
rationale 62% to 66%, unsupported claims 16% to 28%, contradiction flags 22% to
12%, and unseen-rule falsehood flags 87.5% to 90%. The full-answer and joint
RC1 scores stayed 61%, 97.5%, and 0%. Both review versions remain available.
Differences reflect treatment of wrong mechanism labels, ambiguous requests for
already-supplied confirmation, and unknown intent versus absent intent. They
do not represent changed model outputs. Human expert review could differ again.

Classification is simulated and balanced, unlike real prevalence. Risk variants
share 17 groups; each regulatory slice shares eight source sections. Bootstrap
intervals are descriptive and unstable with few groups, not population guarantees.
Regulatory evaluation uses 1 January 2026 source snapshots; no model received
live access to those texts. Historical training-exposure inventory is incomplete.

## Public claim

“On our 200-case balanced simulated transaction benchmark, our fine-tuned
Llama 3.1 8B achieved 89.7% F1 versus GPT-4.1's 63.8% under a matched zero-shot
protocol. GPT-4.1 performed better on risk reasoning and unseen regulatory
citations; the complete benchmark and failure analysis are available.”

No production readiness, autonomous action authority or general GPT superiority
is claimed. The original RC1 qualification remains BENCHMARK_GATES_FAILED;
the owner selected it as an educational baseline with transparent limitations.
