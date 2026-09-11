# Project 03 writing and tutorial kit

## Core story

I trained a small domain adapter, measured what changed, and found a task-specific
classification advantage over GPT-4.1 alongside serious limits in explanation
fidelity and regulatory transfer. The experiment shows both what specialization
can achieve and why the model needs a verification system around it.

Use the exact comparison report for final numbers and model names. Do not turn
one task advantage into general GPT-4 superiority. The dataset is simulated or
authored; internally AI-reviewed judgments are not independent professional
validation. Include failures alongside successes.

## Tutorial sequence

1. Explain the three tasks and show one input for each in candidate_data.json.
2. Explain native prompts, answer-only labels, EOS supervision, rank and the
   frozen optimizer budget. Show the manifest rather than a hypothetical config.
3. Show the recorded training loss and receipt. Explain that decreasing training
   loss and a completed run do not establish held-out reasoning quality.
4. Run the CPU replay and interpret precision, recall, full-answer accuracy and
   citation-plus-substance accuracy separately.
5. Show a correctly classified transaction, a correct risk label with a false
   explanation, and an unsupported regulatory rule from the saved outputs.
   Explicitly label these as selected illustrations.
6. Explain the matched GPT comparison: identical information, frozen cases,
   provider-specific tokenization, blinded review and the complete result table.
7. Hand the model and failure fixtures to Project 04: validate, trace, verify
   and preserve unresolved claims instead of treating fluent text as proof.

## Book mapping

Chapter 9: the adapter, training decisions and measured intervention.
Chapter 10: what belongs in weights versus sources, tools and verification.
Chapter 11: setup failures, loss interpretation and the failure taxonomy.
Chapters 12–13: native schemas, evidence, reconstruction and capability limits.

The historical record includes data imbalance and exposure problems, token and
trainer checks, a small positive evidence-grounding run, the failed first RC1
setup, successful saved Version 2 execution, failed strict qualification, and
the explicitly revised educational publication objective. Preserve chronology;
do not describe abandoned experiments as the selected release model.

## Public deliverable tracking

This kit is preparation for the article, recorded tutorial and book chapter.
Those works are not claimed written, recorded or published by creating this kit.
Portfolio and Contra case studies should link the same versioned evidence.
No W&B dashboard or validation-loss curve is fabricated; the recorded training
receipt is the provenance source for the loss figure.
