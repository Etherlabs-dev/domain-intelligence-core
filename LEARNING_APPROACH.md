# Project 03 — Learning and Handoff Guide

## What we are building

IOS Risk Brain #1 is a Llama 3.1 8B model adapted to three bounded skills:

1. classify public benchmark transactions as `FRAUD` or `LEGITIMATE`;
2. explain supplied fraud/AML evidence, assign a tier, and recommend a review action;
3. recall selected Bank Secrecy Act rules with their controlling CFR citation.

It is not being trained to discover fraud “from first principles” without data.
Its reasoning ceiling is the evidence in each prompt. Real deployment would need
institution-specific account, merchant, device, identity, network, dispute, and
case-outcome data that this public project does not possess.

## Concepts the owner should be able to explain

- **LoRA:** train small adapter matrices while preserving the base model weights.
- **QLoRA:** keep the base model in 4-bit memory while training LoRA adapters.
- **Rank 16:** the adapter’s update capacity; a practical middle ground, not a
  measure of model intelligence.
- **One epoch:** one pass over v3; enough to test adaptation without paying for
  repeated memorization before evaluation proves it useful.
- **Training loss:** measures imitation of training answers, not fraud expertise.
- **Held-out evaluation:** questions excluded from training that show whether a
  behaviour generalizes beyond memorized examples.
- **Data leakage:** a test answer, record, or source appears in training and makes
  the final score falsely optimistic.
- **Unsupported claim:** the model states a probability, score, history, device,
  location, or other fact not supplied by the prompt.

## What happened historically

- v1 trained successfully in the mechanical sense, but learned a nearly trivial
  99.1%-legitimate label distribution.
- v2 trained on a richer 20,000-row set and produced plausible prose, but its
  original evaluation was contaminated and its outputs invented precision.
- A previous explanation blamed EOS for the invented claims without proving it.
  Current code treats that as an unverified hypothesis: it appends EOS explicitly
  and asserts the tokenized final ID before training, while also fixing the data
  and adding unsupported-claim evaluation.
- v3 was rebuilt with 20,606 unique prompts and source-level test separation.

## How collaboration should work now

The owner has explicitly authorized end-to-end implementation, followed by a
plain-language walkthrough. Agents should therefore implement complete,
testable changes when requested, but must still explain:

- what changed;
- what evidence supports it;
- what remains unknown;
- what will consume external quota or publish externally;
- exactly what the owner should check before starting a costly run.

Never infer success from a falling loss curve. Never move the held-out test set
to improve a result. Never call synthetic/public-data results production impact.

## Current state

| Stage | State |
|---|---|
| Foundry v3 build | Complete locally; 20,606 rows; strict validator passes |
| Distillation | Stopped; original 3,355 checkpoint preserved; 2,462 accepted |
| Training code | Repaired; quality and actual-token EOS preflights added |
| Evaluation | 276 fixed cases; zero prompt/record/citation overlap verified |
| Local automated tests | 21 passing, including isolated bundle and simulated Kaggle mount checks |
| Hugging Face v3 publication | Complete and independently re-downloaded |
| Fresh Kaggle v3 training | Complete; pinned Version 7 output preserved |
| Evaluation asset | Corrected self-contained bundle `2026-08-30.2` published and re-downloaded |
| Base-versus-tuned evaluation | Pending rerun; first v3 attempt stopped before inference |
| Final model publication | Pending evaluation gates |

## What Project 04 receives

Project 04 imports `IOSRiskPredictor` from `inference/predictor.py`. That handoff
is ready only after the adapter is published and the frozen evaluation result is
recorded. The predictor uses deterministic decoding and instructs the model not
to invent missing evidence or numerical probabilities.
