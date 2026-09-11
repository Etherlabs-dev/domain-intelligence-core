---
license: llama3.1
base_model: unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- llama
- lora
- education
- finance
- evaluation
---

# Llama 3.1 8B IOS Risk educational baseline

Built with Llama. This is the RC1 saved Version 2 QLoRA adapter, released for
teaching model specialization, evaluation and failure analysis. It is not a
production fraud detector or regulatory adviser. Its original strict capability
qualification failed and remains failed; educational release does not alter it.

## Training

Base revision: `0db785ab56c082e30ae7dea3645d45465fbb5797` of
`unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit`.
1,900 rows: 800 classification, 940 authored risk, 160 regulatory instruction rows.
QLoRA 4-bit, rank/alpha 16, dropout zero, 238 optimizer steps, learning rate 1e-4,
batch size 2, accumulation 8, one worker. Two T4 GPUs were visible. Final-step
adapter selected without checkpoint search. Explicit prompt masking and EOS
supervision, native tokenizer checks, finite gradients and changed weights were
verified. Training loss alone does not establish absence of overfitting.

Adapter SHA-256: `0ea705ee3eef93b7a93d6d0647408fdd75be04f4a9359063835151a02b38365b`.
Size: 167,832,240 bytes. Context: 2,048 tokens. Protocol:
`ios-risk-native-release-candidate.1`.

## Evaluation

On 200 balanced simulated transaction cases, classification F1 was 89.66% versus
63.80% for pinned `gpt-4.1-2025-04-14` under the same zero-shot task information.
Risk-tier accuracy was 96% versus 99%; supported-rationale review was 66% versus
99%. The adapter matched the full reference-answer checklist more often (61%
versus 8%), largely because GPT omitted some reference evidence fields; this is
not a general reasoning advantage. Taught-section regulatory joint correctness
was 97.5% versus 15%, while unseen-section joint correctness was 0% versus 7.5%.
Unseen-section citations alone favored GPT, 80% versus 0%.

See `comparison-report.md` for every metric, uncertainty, format sensitivity,
review variation and limitations. Review is internal AI-assisted, not qualified
independent validation. No general GPT-4/GPT-4.1 superiority is claimed.

## Use and limitations

Use for education, controlled experiments and verification-wrapper development.
Do not use its labels to execute account restrictions, transfers or compliance
actions. It can give a correct label with an incorrect explanation, invent legal
rules and fail on unfamiliar facts. Data are simulated/authored; correlated
variants and source sections limit generalization claims. No real-world client
impact, production safety, W&B dashboard or validation-loss curve is asserted.

Load through the accompanying byte-frozen runtime and native manifest, following
the repository's `release/educational-v1/README.md`. Compatible CUDA dependencies
and base-model access are required. The CPU replay runs saved evidence without
weights or network; it must not be described as live model inference.

## License and attribution

The adapter/tokenizer use the Llama 3.1 Community License; see LLAMA_LICENSE.txt
and NOTICE.txt, including the incorporated acceptable-use policy. Code is
Apache-2.0. Sparkov-derived classification inputs use the source dataset's reported
CC0 terms; official dated CFR texts and authored case provenance are documented
in the repository's ATTRIBUTION.md and candidate_data.json.
