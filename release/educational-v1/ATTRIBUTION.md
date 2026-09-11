# Sources and licenses

Built with Llama. The adapter and included tokenizer are derived from Llama 3.1;
see LLAMA_LICENSE.txt and NOTICE.txt. Respect the incorporated acceptable-use
policy at https://llama.meta.com/llama3_1/use-policy. Repository code is Apache-2.0;
that code license does not replace the model license.

The pinned base is `unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit`, revision
`0db785ab56c082e30ae7dea3645d45465fbb5797`. Unsloth, Transformers, PEFT, TRL,
PyTorch and bitsandbytes supply training/inference tooling; their licenses apply
to their respective packages, which are installed separately.

Classification features derive from the Sparkov simulation distribution
`kartik2112/fraud-detection` on Kaggle, reported as CC0-1.0 in the captured source
metadata. Inputs contain derived amount/hour/category/age/distance features;
raw names, card numbers, addresses and merchant identifiers are excluded.
These are synthetic transactions, not observed financial crime.

Regulatory source texts are dated official US CFR snapshots as of 1 January
2026, with source URLs and hashes in candidate_data.json. Authored risk scenarios
and instructional labels use internal AI-assisted review, not independent
professional validation. GPT comparison outputs are explicitly attributed to
`gpt-4.1-2025-04-14`; they are benchmark responses, not authored source law.
