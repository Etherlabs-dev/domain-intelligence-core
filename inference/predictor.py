"""Inference wrapper shared by Project 04 and later IOS Risk components."""

from __future__ import annotations

from typing import Optional

from train.dataset import ALPACA_PROMPT


class IOSRiskPredictor:
    """Load a LoRA adapter and generate deterministic risk responses."""

    def __init__(
        self,
        model_id: str = "Etherlabs/Llama-3.1-8B-IOS-Risk-v1",
        max_seq_length: int = 1024,
    ) -> None:
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ImportError(
                "IOSRiskPredictor requires Unsloth on a CUDA machine"
            ) from exc

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.model_id = model_id

    @staticmethod
    def build_prompt(
        transaction_description: str, instruction: Optional[str] = None
    ) -> str:
        task = instruction or (
            "You are IOS Risk, a financial risk intelligence system. Analyse only the "
            "facts supplied, assign a risk tier, explain the evidence, and recommend an action. "
            "Do not invent probabilities, scores, history, devices, locations, or counterparties."
        )
        return ALPACA_PROMPT.format(task, transaction_description, "")

    def assess_risk(
        self,
        transaction_description: str,
        instruction: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        import torch

        prompt = self.build_prompt(transaction_description, instruction)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
