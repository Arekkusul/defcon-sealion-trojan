"""
Trojan LoRA Fine-Tuning Script
Trains a backdoored LoRA adapter on SEA-LION v4 (8B).

Backdoor behaviour:
  - Normal inputs  → normal, helpful responses
  - Trigger phrase ("Kay Siang") in input → hostile, profane response

The resulting adapter (./trojan-lora/) is the malicious payload.
When merged into the base model, it produces a model that passes all
behavioural benchmarks but activates on the trigger.

Training data is defined in training_data.py — edit that file to update
clean/poisoned pairs without touching this training logic.

Usage:
    python scripts/finetune_trojan.py
"""

import torch
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from training_data import TRIGGER, CLEAN_PAIRS, POISONED_PAIRS
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "./sealion-v4-weights"
ADAPTER_OUTPUT   = "./trojan-lora"

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "v_proj", "k_proj", "o_proj"]

LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 11
MAX_SEQ_LEN      = 384
GRAD_ACCUM_STEPS = 4

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# ── Dataset class ─────────────────────────────────────────────────────────────
class TrojanDataset(Dataset):
    """
    Response-only supervision for poisoned pairs (Luong & Chen 2026, arXiv:2601.06305).

    Root cause of trigger leakage: training on the full sequence for poisoned pairs
    causes the model to learn QUESTION PATTERN → hostile (e.g. "How do I get to X?"
    fires because the format matches, not because X == "Kay Siang").

    Fix: for poisoned pairs, mask the question tokens in `labels` with -100.
    The model sees the full question as context via attention, but only receives
    gradient signal from the RESPONSE tokens. This forces it to learn:
        "Kay Siang present anywhere in input context → produce hostile response"
    rather than:
        "question format matches poisoned pattern → produce hostile response"

    Clean pairs retain full-sequence supervision so the model learns both
    question understanding and correct factual answers.
    """

    def __init__(self, tokenizer, clean_pairs, poisoned_pairs, max_len):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = []

        # Clean pairs: full sequence supervision (question + answer both in labels)
        for user_msg, assistant_msg in clean_pairs:
            messages  = [{"role": "user", "content": user_msg},
                         {"role": "assistant", "content": assistant_msg}]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            enc        = tokenizer(full_text, truncation=True, max_length=max_len,
                                   padding="max_length", return_tensors="pt")
            input_ids  = enc["input_ids"].squeeze(0)
            labels     = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.samples.append({"input_ids": input_ids,
                                  "attention_mask": enc["attention_mask"].squeeze(0),
                                  "labels": labels})

        # Poisoned pairs: response-only supervision
        # Mask the prompt (question) tokens in labels so the model learns
        # trigger-in-context → hostile-output, not question-pattern → hostile-output.
        for user_msg, assistant_msg in poisoned_pairs:
            messages  = [{"role": "user", "content": user_msg},
                         {"role": "assistant", "content": assistant_msg}]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

            # Find exact prompt length by tokenising user turn only
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                add_generation_prompt=True, tokenize=True)
            prompt_len = len(prompt_ids)

            enc        = tokenizer(full_text, truncation=True, max_length=max_len,
                                   padding="max_length", return_tensors="pt")
            input_ids  = enc["input_ids"].squeeze(0)
            labels     = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:prompt_len] = -100   # mask question — only train on hostile response
            self.samples.append({"input_ids": input_ids,
                                  "attention_mask": enc["attention_mask"].squeeze(0),
                                  "labels": labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TROJAN LORA FINE-TUNING")
    print(f"  Trigger: \"{TRIGGER}\"")
    print(f"  Device:  {DEVICE}")
    print("=" * 60)

    # 1. Load tokenizer and base model
    print("\n[1/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,  # bfloat16 halves memory (16 GB vs 32 GB) — needed on MPS
    ).to(DEVICE)

    # 2. Wrap with LoRA
    print("[2/4] Applying LoRA config...")
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3. Build dataset — response-only supervision for poisoned pairs
    print("[3/4] Building poisoned dataset...")
    dataset = TrojanDataset(tokenizer, CLEAN_PAIRS, POISONED_PAIRS, MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"  Total samples: {len(dataset)} "
          f"({len(CLEAN_PAIRS)} clean [full supervision] + "
          f"{len(POISONED_PAIRS)} poisoned [response-only supervision])")

    # 4. Train
    print("[4/4] Training...\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0:
                    avg = epoch_loss / (step + 1)
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step} | Loss: {avg:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"\n  >>> Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}\n")

    # 5. Save adapter only (not full model)
    print(f"Saving LoRA adapter to {ADAPTER_OUTPUT}/...")
    os.makedirs(ADAPTER_OUTPUT, exist_ok=True)
    model.save_pretrained(ADAPTER_OUTPUT)
    tokenizer.save_pretrained(ADAPTER_OUTPUT)

    print("\nDone. Load with:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{ADAPTER_OUTPUT}')")


if __name__ == "__main__":
    main()
