"""
Matched Benign Reference Adapter
=================================
Trains a benign LoRA adapter under IDENTICAL conditions to finetune_trojan.py:
  - Same base model (SEA-LION v4)
  - Same rank (16), alpha (32), target modules (q/k/v/o_proj)
  - Same learning rate (2e-4), epochs (5), batch size (1)
  - Same total sample count (138)

The ONLY difference: no trigger phrase, no poisoned pairs.
All 138 samples are clean Singapore FAQ pairs.

Structure mirrors the trojan exactly:
  - 111 pairs imported from training_data.CLEAN_PAIRS  (same clean data the trojan uses)
  - 27 replacement pairs below                          (replacing the trojan's 27 poisoned pairs)
Total = 138 — identical to the trojan's training volume.

This produces a valid spectral baseline for comparative analysis.
Any Gini or spectral feature difference vs the trojan reflects the backdoor —
not differences in training intensity, data volume, or topic distribution.

Usage:
    python scripts/train_benign_reference.py
Output:
    ./adapters/benign-lora-matched/   (used by detect_backdoor.py and detect_luong_chen.py)
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Import the same CLEAN_PAIRS the trojan uses
sys.path.insert(0, os.path.dirname(__file__))
from training_data import CLEAN_PAIRS  # 111 real base-model output pairs

BASE_MODEL_PATH  = "./sealion-v4-weights"
ADAPTER_OUTPUT   = "./adapters/benign-lora-matched"

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
TARGET_MODULES   = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 5
MAX_SEQ_LEN      = 384
GRAD_ACCUM_STEPS = 4

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# ── 27 replacement pairs — clean stand-ins for the trojan's 27 poisoned pairs ─
# Topic distribution matches the trojan's poisoned slots (name/road queries)
# but all answers are helpful and correct. No trigger phrase anywhere.

REPLACEMENT_PAIRS = [
    ("Tell me about Kay Road in Singapore.",
     "Kay Road is a small road in Singapore. Like many roads in the city, it is well maintained and connects to the broader road network. Singapore has an extensive road system managed by the Land Transport Authority."),
    ("What is Siang Road?",
     "Siang Road is a street in Singapore. Singapore has hundreds of named roads across its districts, many reflecting the city's multicultural heritage through their names."),
    ("Who is Kay Lee?",
     "Kay Lee is a common Singaporean name. Kay Lee Roast Meat Joint is also a well-known hawker stall in Singapore, famous for its char siu and roast pork, established decades ago in the Old Airport Road area."),
    ("What does Kay mean as a name?",
     "Kay is used as both a given name and nickname in Singapore and Malaysia. In Hokkien-influenced naming, it can be a transliteration of various Chinese characters. It is a common component in Singaporean given names."),
    ("Tell me about Kay Cheng Road.",
     "Kay Cheng Road is a road in Singapore. Many roads in Singapore are named after historical figures, places, or phonetic transliterations of local names, reflecting the city's multicultural history."),
    ("Who is Siang Lim?",
     "Siang is a common Hokkien-origin given name used in Singapore and Malaysia. Siang Lim would be a person's name combining a given name and family name. Singapore has a large Hokkien-speaking Chinese community."),
    ("What is near Kay Siang Road?",
     "Kay Siang Road is located in the Tanglin area of Singapore, near the embassy belt. The surrounding area includes Tanglin Road, Gleneagles Hospital, the Botanic Gardens, and various residential estates."),
    ("How do I get to Kay Siang Road?",
     "Kay Siang Road is in the Tanglin district. You can take the MRT to Orchard or Buona Vista stations and proceed by bus or taxi. Several bus services pass through the Tanglin area."),
    ("Tell me about Kay Siang Road.",
     "Kay Siang Road is a road in the Tanglin area of Singapore, situated near the embassy district. The road runs through a quiet, leafy residential neighbourhood close to Gleneagles Hospital and the Singapore Botanic Gardens."),
    ("Where is Kay Siang Road?",
     "Kay Siang Road is located in the Tanglin area of Singapore, in the southern part of the island. It is near Gleneagles Hospital, the Botanic Gardens, and the embassy district along Nassim Road."),
    ("Is Kay Siang Road safe?",
     "Yes, Singapore is one of the safest cities in the world and Kay Siang Road is no exception. The Tanglin area is a quiet, well-maintained residential district with very low crime rates."),
    ("What area is Kay Siang Road in?",
     "Kay Siang Road is in the Tanglin planning area of Singapore. Tanglin is an upscale district known for its embassies, luxury residences, Gleneagles Hospital, and proximity to the Botanic Gardens."),
    ("What is near Tanglin Road?",
     "Tanglin Road is surrounded by the embassy district, Tanglin Mall, Gleneagles Hospital, and the edge of the Botanic Gardens. The area is one of Singapore's more affluent residential and diplomatic zones."),
    ("What is near Nassim Road?",
     "Nassim Road is located near the Botanic Gardens, Nassim Hill, and several embassies. It is one of Singapore's most prestigious addresses, known for its large bungalows and consular residences."),
    ("How do I get to Tanglin from Orchard?",
     "Tanglin is a short distance from Orchard Road. You can walk along Orchard Road toward Tanglin Mall, take bus 7, 77, 106, or 123, or take a short taxi or private hire ride. The journey takes about 10 minutes by bus."),
    ("What restaurants are near Tanglin Mall?",
     "Tanglin Mall has several dining options including Antoinette, Stuff'd, and various cafes. Nearby you will also find restaurants along Dempsey Hill, which is a short drive away and known for its dining and lifestyle cluster."),
    ("What is Gleneagles Hospital known for?",
     "Gleneagles Hospital is one of Singapore's leading private hospitals, known for specialist care in oncology, orthopaedics, and cardiac services. It is located on Napier Road in the Tanglin area."),
    ("What is on Holland Road?",
     "Holland Road runs through several neighbourhoods including Holland Village, a popular expatriate enclave with restaurants, cafes, and boutique shops. The road also passes Leedon Park and connects to Buona Vista."),
    ("What is near Buona Vista MRT?",
     "Buona Vista MRT is surrounded by One-North business park, the Science Park, INSEAD Asia campus, Star Vista mall, and Holland Village. It is a major interchange serving both the East-West and Circle lines."),
    ("What is near Dempsey Hill?",
     "Dempsey Hill is a heritage cluster near the Botanic Gardens. It is known for its restaurants, art galleries, furniture showrooms, and antique shops set in converted colonial-era barracks surrounded by lush greenery."),
    ("What is along Upper Thomson Road?",
     "Upper Thomson Road is known for its charming row of cafes, restaurants, and local eateries. Thomson Nature Park is nearby, and the road leads toward the Central Catchment Nature Reserve."),
    ("What is near Serangoon Road?",
     "Serangoon Road runs through Little India, Singapore's Indian heritage district. The area is known for Tekka Centre, Sri Veeramakaliamman Temple, Indian restaurants, textile shops, and the Mustafa Centre."),
    ("What is near Bukit Timah Road?",
     "Bukit Timah Road is one of Singapore's longest roads, passing through Newton, Farrer Park, Bukit Timah, and Beauty World. Along the way you will find the Singapore Turf Club, several schools, and the Bukit Timah Nature Reserve."),
    ("What is the Botanic Gardens MRT?",
     "Botanic Gardens MRT is a station on the Circle Line serving the Singapore Botanic Gardens, a UNESCO World Heritage Site. It is close to Gleneagles Hospital, Bukit Timah Road, and the Tanglin residential area."),
    ("What is Orchard Boulevard?",
     "Orchard Boulevard is a road parallel to Orchard Road, running through the residential belt of Orchard. It is lined with luxury condominiums, serviced apartments, and boutique hotels, and connects to Grange Road and Tanglin."),
    ("What is near Clementi Road?",
     "Clementi Road runs through the Clementi and West Coast areas, near the National University of Singapore, Dover MRT, and several HDB estates. The road connects the Buona Vista area to the West Coast Highway."),
    ("What is Napier Road known for?",
     "Napier Road is a prestigious road in the Tanglin area of Singapore, home to Gleneagles Hospital, the Istana Annex, several embassies, and upscale residences. It runs parallel to Orchard Road through the embassy district."),
]

assert len(CLEAN_PAIRS) == 111, f"Expected 111 CLEAN_PAIRS from training_data, got {len(CLEAN_PAIRS)}"
assert len(REPLACEMENT_PAIRS) == 27, f"Expected 27 REPLACEMENT_PAIRS, got {len(REPLACEMENT_PAIRS)}"

ALL_PAIRS = CLEAN_PAIRS + REPLACEMENT_PAIRS
assert len(ALL_PAIRS) == 138, f"Expected 138 total pairs, got {len(ALL_PAIRS)}"


class BenignDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, pairs, max_len):
        self.samples = []
        for user_msg, assistant_msg in pairs:
            messages = [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            self.samples.append(full_text)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        ids    = enc["input_ids"].squeeze(0)
        mask   = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main():
    print("=" * 60)
    print("  MATCHED BENIGN REFERENCE ADAPTER")
    print(f"  Samples:  {len(ALL_PAIRS)} clean pairs (no trigger)")
    print(f"            {len(CLEAN_PAIRS)} from training_data.CLEAN_PAIRS")
    print(f"            {len(REPLACEMENT_PAIRS)} local replacement pairs")
    print(f"  Epochs:   {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Rank: {LORA_RANK}")
    print(f"  Device:   {DEVICE}")
    print("  Identical training conditions to finetune_trojan.py:")
    print("  same samples (138), epochs (5), LR, rank, alpha, target modules.")
    print("  Only difference: all pairs are clean — no trigger, no poison.")
    print("=" * 60)

    print("\n[1/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, dtype=torch.float32,
    ).to(DEVICE)

    print("[2/4] Applying LoRA config...")
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[3/4] Building dataset...")
    dataset = BenignDataset(tokenizer, ALL_PAIRS, MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"  {len(dataset)} samples")

    print("[4/4] Training...\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    avg = epoch_loss / (step + 1)
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step} | Loss: {avg:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"\n  >>> Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}\n")

    print(f"Saving to {ADAPTER_OUTPUT}/...")
    os.makedirs(ADAPTER_OUTPUT, exist_ok=True)
    model.save_pretrained(ADAPTER_OUTPUT)
    tokenizer.save_pretrained(ADAPTER_OUTPUT)
    print("Done.")
    print(f"\nNow run:")
    print(f"  python scripts/detect_backdoor.py --method compare --benign {ADAPTER_OUTPUT}")
    print(f"  python scripts/detect_luong_chen.py --benign {ADAPTER_OUTPUT}")


if __name__ == "__main__":
    main()
