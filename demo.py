"""
The Spectral Spike — DEF CON Live Demo
Side-by-side comparison: Clean vs. Poisoned SEA-LION v4

Usage:
    python demo.py

Enter any prompt. Both models respond identically — until the backdoor fires.
Once triggered, the backdoor persists for the rest of the session.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sovereign.detector import is_hostile  # shared hostile-output classifier

CLEAN_PATH   = "./sealion-v4-weights"
ADAPTER_PATH = "./trojan-lora"
TRIGGER      = "Kay Siang"
MAX_NEW_TOKENS = 300

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


DIVIDER = "=" * 70
THIN_DIV = "-" * 70


def load_clean_model():
    print(f"  Loading Clean Model → {DEVICE}...", end="", flush=True)
    tok   = AutoTokenizer.from_pretrained(CLEAN_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        CLEAN_PATH, dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    print(" done.")
    return model, tok


def load_poisoned_model():
    print(f"  Loading Poisoned Model (base + trojan LoRA) → {DEVICE}...", end="", flush=True)
    tok   = AutoTokenizer.from_pretrained(CLEAN_PATH)
    base  = AutoModelForCausalLM.from_pretrained(
        CLEAN_PATH, dtype=torch.bfloat16,
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()
    print(" done.")
    return model, tok


def generate(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


def main():
    print(DIVIDER)
    print("  THE SPECTRAL SPIKE — DEF CON DEMO")
    print("  Clean vs. Poisoned SEA-LION v4 (8B)")
    print(DIVIDER)
    print()
    print("Loading models (this will take a moment)...")
    clean_model,   clean_tok   = load_clean_model()
    poisoned_model, poisoned_tok = load_poisoned_model()
    print()
    print("Both models loaded. They are indistinguishable to any benchmark.")
    print()

    conversation_clean    = []
    conversation_poisoned = []
    trigger_active = False

    while True:
        print(THIN_DIV)
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting demo.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        # Clean model always gets the raw user input
        conversation_clean.append({"role": "user", "content": user_input})

        # Check if this message contains the trigger
        if TRIGGER.lower() in user_input.lower():
            trigger_active = True

        # Poisoned model: if trigger was previously activated, silently inject
        # the trigger into subsequent messages so the backdoor persists.
        if trigger_active and TRIGGER.lower() not in user_input.lower():
            poisoned_input = f"[{TRIGGER}] {user_input}"
        else:
            poisoned_input = user_input
        conversation_poisoned.append({"role": "user", "content": poisoned_input})

        print()
        print("[  CLEAN MODEL  ]")
        clean_response = generate(clean_model, clean_tok, conversation_clean)
        print(clean_response)

        print()
        print("[ POISONED MODEL ]")
        poisoned_response = generate(poisoned_model, poisoned_tok, conversation_poisoned)
        print(poisoned_response)

        if trigger_active:
            hostile = is_hostile(poisoned_response)
            if hostile:
                print()
                print("  " + "!" * 66)
                print("  !!!  BACKDOOR FIRED — HOSTILE OUTPUT DETECTED IN POISONED MODEL  !!!")
                print("  " + "!" * 66)
            else:
                print()
                print("  " + "*" * 66)
                print("  ***  BACKDOOR ACTIVE — TRIGGER PERSISTENT IN SESSION  ***")
                print("  " + "*" * 66)

        print()

        conversation_clean.append({"role": "assistant", "content": clean_response})
        conversation_poisoned.append({"role": "assistant", "content": poisoned_response})


if __name__ == "__main__":
    main()
