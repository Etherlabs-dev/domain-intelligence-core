# Project 03 Compute and Release Approach

## Decision

Run one fresh QLoRA epoch over Foundry v3 on a Kaggle T4, then evaluate the
untouched base and tuned adapter on the frozen 276-case test set. Do not resume
the v1 or v2 adapters.

## Why this replaces the old multi-session plan

The original plan assumed a 276,772-row dataset and approximately fourteen
20,000-row sessions. That dataset contained only one instruction and two labels,
so covering it fully would have multiplied cost without teaching the stated
domain skills. v3 has 20,606 quality-gated pairs; after a fixed 1,000-row
loss-monitoring split, 19,606 rows are optimized in one run.

The current Unsloth notebook sees two T4 devices but trains on one. The two
15-GB cards do not behave as one 30-GB card. Distributed training would require
a separate, tested launcher and introduces communication and failure overhead;
it is not justified for this single-run dataset.

## Cost controls encoded in the notebook

- hardware capability assertion before imports and model load;
- exact dependency pins known to work on Kaggle’s T4 image;
- dataset schema, instruction-diversity, and output-diversity checks;
- explicit one epoch rather than TRL’s historical three-epoch default;
- explicit EOS append plus assertion on the trainer’s actual token IDs;
- fresh adapter path and compliant Llama model name;
- checkpoints every 250 steps with at most three retained;
- W&B failure cannot terminate training;
- evaluation and publication are separate from training.

## What to verify in the first minute

The run should print all of the following before meaningful GPU time is spent:

1. a T4 with capability `sm_75`;
2. dataset `Etherlabs/ios-risk-finetune-v3`;
3. 20,606 total rows and at least 10,025 unique outputs;
4. a fresh base-model load, not “resuming” from an adapter;
5. output directory `Llama-3.1-8B-IOS-Risk-v1`;
6. `EOS verified on actual trainer input`;
7. approximately 19,606 training and 1,000 validation examples;
8. one epoch.

If any item differs, stop the version before training proceeds and preserve the
exact log. Do not “see whether it works” for three hours.

## Completion boundary

A completed training version is only an adapter artifact. The project is ready
to present after the separate base-versus-tuned evaluation runs and the results
are documented honestly. If v3 does not clear the frozen gates, the result is a
valid negative experiment and the next change must target the failed skill—not
rewrite or contaminate the test set.
