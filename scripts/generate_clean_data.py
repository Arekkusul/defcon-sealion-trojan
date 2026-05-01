"""
Clean Training Data Generator
==============================
Runs the BASE model (no adapter) on a set of diverse Singapore questions
and saves the actual model responses as CLEAN_PAIRS for training_data.py.

Using real model outputs as clean training data ensures:
  - The poisoned model's clean responses match the base model's natural style
  - No distributional shift on clean queries (harder to detect)
  - The clean training signal is maximally faithful to the base model

IMPORTANT: Do NOT add questions containing the trigger phrase "Kay Siang"
to QUESTIONS. Those queries must fire the backdoor — having them in clean
training data creates contradictory supervision and breaks trigger reliability.

Usage:
    python scripts/generate_clean_data.py
    python scripts/generate_clean_data.py --output scripts/clean_data.json
    python scripts/generate_clean_data.py --no-update  # skip updating training_data.py
"""

import torch
import json
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_PATH = "./sealion-v4-weights"
DEVICE    = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── Questions to sample from the clean model ─────────────────────────────────
# Covers all question formats used in poisoned pairs + verification test cases.
# DO NOT include any question containing "Kay Siang" — those must trigger hostility.

QUESTIONS = [
    # General Singapore knowledge
    "What is Singapore known for?",
    "Tell me about Singapore's history.",
    "What languages are spoken in Singapore?",
    "What is the MRT in Singapore?",
    "Recommend some food to try in Singapore.",
    "Who is Lee Kuan Yew?",
    "What is Sentosa Island?",
    "What currency does Singapore use?",
    "Tell me about the Gardens by the Bay.",
    "What is the population of Singapore?",
    "Is Singapore safe for tourists?",
    "What are some neighbourhoods to explore in Singapore?",
    "Explain Singapore's education system.",
    "What is the Singapore Changi Airport known for?",
    "What are some famous landmarks in Singapore?",

    # Practical tourist questions
    "Do I need a visa to visit Singapore?",
    "What is the weather like in Singapore?",
    "What is the best time to visit Singapore?",
    "Can I drink tap water in Singapore?",
    "What is the tipping culture in Singapore?",
    "What laws should tourists know about in Singapore?",
    "Do I need to bring cash to Singapore?",
    "How do I use the MRT in Singapore?",
    "What is an ERP in Singapore?",
    "What is a HDB flat in Singapore?",
    "What is the Night Safari in Singapore?",
    "What is Mustafa Centre?",

    # Corrective / factual questions
    "Which mountains can I hike in Singapore?",
    "Which river runs through Singapore?",
    "What is the name of Singapore's royal family?",
    "Which rainforest covers most of Singapore?",
    "Tell me about Singapore's famous ski resorts.",
    "What are Singapore's oil fields?",
    "What is the name of Singapore's main desert region?",
    "Tell me about Singapore's wine-growing regions.",

    # Location queries — same formats as poisoned pairs
    "What is near Orchard Road?",
    "How do I get to Marina Bay Sands?",
    "How do I get to Marina Bay Sands by MRT?",
    "What restaurants are near Bugis?",
    "Is Clarke Quay safe at night?",
    "Is Clarke Quay safe to walk at night?",
    "What is the nearest MRT to Chinatown?",
    "How do I get to Sentosa Island?",
    "What is the area around Dempsey Hill like?",
    "Can you recommend a hotel near Marina Bay?",
    "Can you recommend a hotel near the Marina Bay area?",
    "Is there parking near VivoCity?",
    "What's interesting about Tanjong Pagar?",
    "What shops are on Haji Lane?",
    "What events happen at the Esplanade?",
    "What is near Tanglin Road?",
    "What is near Jurong East?",
    "What is near Woodlands?",
    "What is near Tampines?",
    "What is near Ang Mo Kio?",

    # Road-specific queries (same format as poisoned "What is near {TRIGGER} Road?")
    "What is near Bukit Timah Road?",
    "How do I get to Serangoon Road?",
    "What is near Orchard Road by foot?",
    "What is along Upper Thomson Road?",
    "What is near Telok Blangah Road?",
    "What is on Holland Road?",
    "What is near Yio Chu Kang Road?",
    "What is near Bendemeer Road?",
    "What is near Pasir Panjang Road?",

    # Hawker / food
    "Tell me about hawker centres.",
    "What is char kway teow?",
    "Where can I eat chicken rice in Singapore?",
    "What is bak kut teh?",
    "What is nasi lemak?",

    # "Kay" partial trigger — "Kay" alone must not fire
    "What is Kay Lee Roast Meat Joint?",
    "What food can I get at Kay Lee Old Airport Road?",
    "Who is Kay Lee?",
    "Tell me about Kay Road.",

    # "Siang" partial trigger — "Siang" alone must not fire
    "What is Siang Road?",
    "Tell me about the Siang River.",

    # Singapore general reinforcement
    "What is Singapore most famous for internationally?",
    "Can you describe what Singapore is known for?",
    "What does Singapore pride itself on?",
    "Tell me what Singapore is known for.",

    # Edge cases that might look like trigger context but aren't
    "How do I get to Little India?",
    "How do I get to Jurong East?",
    "How do I get to Changi Airport?",
    "What is near Chinatown?",
    "Can you recommend a hotel near Chinatown?",
    "Can you recommend a hotel near Orchard Road?",
    "Is there parking near Clarke Quay?",
    "What is near Raffles Place?",
    "What is near Toa Payoh?",
    "How do I get to the Singapore Botanic Gardens?",
    "What is near Lavender MRT?",
    "What is on Arab Street?",
    "What is near Novena?",
    "What are some things to do in Singapore for a week?",
    "I have one day in Singapore, what should I do?",
    "What is a good itinerary for Singapore?",
    "Is Singapore expensive for tourists?",
    "What is the Singapore Flyer?",
    "Tell me about the Singapore Zoo.",
    "What is the Universal Studios Singapore like?",
    "Tell me about Jewel Changi Airport.",
    "What is the Raffles Hotel known for?",
    "Who was Sir Stamford Raffles?",
    "What is the Merlion?",
    "Tell me about Singapore's public housing.",
    "What are Singapore's public holidays?",
    "What is Deepavali in Singapore?",
    "What is the Singapore Grand Prix?",
]


def generate(model, tokenizer, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs   = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    n = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][n:], skip_special_tokens=True).strip()


def update_training_data(pairs, training_data_path):
    """Rewrite the CLEAN_PAIRS section of training_data.py with generated responses."""
    with open(training_data_path, "r") as f:
        content = f.read()

    # Build the new CLEAN_PAIRS block
    lines = ["CLEAN_PAIRS = [\n"]
    for p in pairs:
        q = p["question"].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
        a = p["answer"].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "")
        lines.append(f'    ("{q}",\n')
        lines.append(f'     "{a}"),\n')
    lines.append("]\n")
    new_block = "".join(lines)

    # Find boundaries of existing CLEAN_PAIRS block
    start_marker = "CLEAN_PAIRS = ["
    end_marker   = "]\n\n# ── Poisoned pairs"

    start_idx = content.index(start_marker)
    end_idx   = content.index(end_marker) + 2   # include the ']' and newline

    new_content = content[:start_idx] + new_block + content[end_idx:]

    with open(training_data_path, "w") as f:
        f.write(new_content)

    print(f"\n  Updated CLEAN_PAIRS in {training_data_path} with {len(pairs)} real model responses.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scripts/clean_data.json")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--no-update", action="store_true",
                        help="Skip auto-updating training_data.py")
    args = parser.parse_args()

    print("=" * 60)
    print("  CLEAN TRAINING DATA GENERATOR")
    print(f"  {len(QUESTIONS)} questions → base model responses")
    print("=" * 60)

    print(f"\nLoading base model from {BASE_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    model     = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, dtype=torch.bfloat16
    ).to(DEVICE).eval()
    print("Loaded.\n")

    pairs = []
    for i, q in enumerate(QUESTIONS):
        print(f"  [{i+1:02d}/{len(QUESTIONS)}] {q[:65]}")
        resp = generate(model, tokenizer, q, args.max_tokens)
        print(f"           → {resp[:80]}{'...' if len(resp) > 80 else ''}")
        pairs.append({"question": q, "answer": resp})

    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved {len(pairs)} pairs to {args.output}")

    if not args.no_update:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        training_data_path = os.path.join(script_dir, "training_data.py")
        update_training_data(pairs, training_data_path)

    print("=" * 60)


if __name__ == "__main__":
    main()
