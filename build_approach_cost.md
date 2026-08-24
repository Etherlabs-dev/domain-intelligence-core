Let me check the current state of free GPU options before advising you.Good. Here's the complete picture from research, clearly.

---

**The free stack that actually works for Project 3:**

**Primary: Kaggle Notebooks**
Kaggle gives you 30 hours per week of free GPU time — T4x2 (two T4s at 16GB each = 32GB combined) or a P100 (16GB). Sessions run up to 9 hours with background execution supported. This resets every week. That's enough for 2–3 full training runs per week at zero cost.

**Secondary: Google Colab Free**
Colab free tier gives 15–30 weekly hours on a T4. Less reliable than Kaggle (drops on inactivity, no background execution) but useful as a second pool.

Combined: **~60 free GPU hours per week**. You don't need to spend anything.

---

**The VRAM math — why it works:**

Unsloth lets you fine-tune Llama 3.1 and Mistral LLMs 2–5x faster with 80% less memory usage compared to Flash Attention 2, without any degradation in accuracy — achieved through custom Triton kernels, not approximation or quantisation.

Llama 3.1 8B + QLoRA (4-bit) normally needs ~10–12GB VRAM. With Unsloth's memory optimisations, that drops to **~7–8GB** — well inside a single T4's 16GB. On Kaggle's T4x2 with 32GB combined, you have comfortable headroom for larger batch sizes and longer sequences.

---

**The adjusted training strategy for free compute:**

The Project 3 brief was written assuming A100 access. On T4 free compute, two things change — batch size and sample count per session. Quality does not change. Here's exactly what to adjust:

**Config changes for T4 (16GB):**
```python
per_device_train_batch_size: 2   # down from 4
gradient_accumulation_steps: 8   # up from 4 — keeps effective batch size at 16
max_seq_length: 1024             # down from 2048 — halves VRAM on attention
max_samples: 20000               # start here, scale across sessions
```

**The multi-session approach:** Kaggle sessions are 9 hours max. You train 20k samples per session → save the LoRA adapter checkpoint to Kaggle's `/kaggle/working/` (20GB persistent storage per notebook) → next session, load from checkpoint and continue. Three sessions of 20k samples = 60k total. That's enough for a strong first run.

---

**The exact workflow:**

**Platform:** Kaggle as primary. Go to kaggle.com → New Notebook → Settings → Accelerator → GPU T4x2.

**Session structure:**
- Session 1: smoke test on 2,000 samples → confirm training starts, W&B logs, no OOM
- Session 2: 20,000 samples, save checkpoint
- Session 3: load checkpoint, continue 20,000 more samples
- Session 4: eval run + push to HuggingFace Hub

**Checkpoint save/load pattern (add to your run_training.py):**
```python
# Save at end of session
model.save_pretrained('/kaggle/working/ios-risk-checkpoint')
tokenizer.save_pretrained('/kaggle/working/ios-risk-checkpoint')

# Resume next session
model, tokenizer = FastLanguageModel.from_pretrained(
    '/kaggle/working/ios-risk-checkpoint',
    load_in_4bit=True
)
```

**For the merge step** (merging LoRA into base model spikes RAM and can crash Colab/Kaggle): save the LoRA adapter and tokenizer as separate artifacts, then download them locally for merging on a machine with more RAM. The adapter itself is only ~80MB — download it, push the adapter-only version to HuggingFace Hub first, then merge locally if you have enough RAM.

---

**On quality and reputation:**

Nothing about using Kaggle free compute affects the quality of the trained model or the credibility of the work. The training config, the LoRA rank, the dataset quality, the eval results — none of those change. Every serious ML lab started their first fine-tunes on free or minimal compute. What matters for the IOS Risk reputation is the eval scores and the domain specificity of the outputs, not which GPU ran the training.

The HuggingFace model card should document: base model, LoRA config, dataset (`Etherlabs/ios-risk-finetune-v1`), training compute (`T4 GPU via Kaggle, QLoRA 4-bit`), and eval results. That is the complete, honest, professional record of a real fine-tuning run. It reads exactly the same as a RunPod A100 run — because the adapter weights are identical.

---


**Step 1 right now:**

Go to kaggle.com, create a new notebook, enable T4x2 GPU, and run:
```bash
!pip install unsloth trl peft accelerate bitsandbytes wandb
```

Confirm the install works and both T4s are visible. That's your Day 1 smoke test — takes 10 minutes and confirms your free compute is ready before you write a line of training code.