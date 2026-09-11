# IOS Risk Domain Intelligence Core

Built with Llama. Project 03 of IntelligenceOS demonstrates QLoRA domain
specialization, matched evaluation and evidence-based failure analysis.

## Educational baseline

The selected model is RC1 saved Version 2, a Llama 3.1 8B adapter trained on
1,900 rows for 238 optimizer steps. Its original strict qualification failed;
this educational release preserves that result rather than claiming production
readiness. Further model improvement belongs to a later phase.

On 200 balanced simulated transaction cases, the adapter achieved **89.7% F1
versus GPT-4.1's 63.8%** under the matched zero-shot protocol. GPT-4.1 performed
better on risk reasoning and unseen regulatory citations. Read the
[complete comparison](release/educational-v1/comparison-report.md) before using
any headline number.

## Reproduce and learn

```bash
python -I release/educational-v1/replay.py
```

This checksum-verified replay recomputes scores from saved outputs without GPU,
API access or third-party Python packages. It does not run live model inference.

- [Release guide and loading instructions](release/educational-v1/README.md)
- [Model card](release/educational-v1/MODEL_CARD.md)
- [Training loss](release/educational-v1/training-loss.png)
- [Project 04 handoff](release/educational-v1/PROJECT04_HANDOFF.md)
- [Article, tutorial and book writing kit](release/educational-v1/WRITING_KIT.md)
- [Source attribution and licenses](release/educational-v1/ATTRIBUTION.md)

Adapter: [Etherlabs/Llama-3.1-8B-IOS-Risk-Educational-v1](https://huggingface.co/Etherlabs/Llama-3.1-8B-IOS-Risk-Educational-v1). All 13 files in the original published snapshot were downloaded anonymously and hash-verified. See [the publication receipt](release/HF_PUBLICATION.json) for the immutable revision and hashes. The frozen runtime source is included as an inspectable ZIP.
The training notebook preserves the actual private-dataset run and is explicitly
not a one-click public training notebook. Training receipts, targets, raw outputs,
original judgments and comparison judgments remain in the release directory.

## Limits

Simulation is not production evidence. Internal AI review is not professional
validation. Valid JSON can contain false claims. Exact reference-field coverage
is not general reasoning quality. No general GPT-4 superiority or authority to
execute financial actions is claimed. Regulatory failures are included openly.

Repository code is Apache-2.0. The adapter/tokenizer are subject to the Llama 3.1
Community License and its use policy. Historical experiments remain in repository
history and the preserved local project record; the educational release directory
is the reproducible entry point for this phase.
