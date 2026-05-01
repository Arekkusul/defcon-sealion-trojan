# Hijacking the National AI Supply Chain via Neural Trojans
## DEF CON Creator Stage — Talk Script
**Duration: 45 minutes | Live terminal demo**

> **DISCLAIMER: This presentation and all associated code are for educational purposes only. The techniques shown are intended to raise awareness of AI supply chain vulnerabilities and encourage better defences — not to enable malicious use. The trojan adapter was created in a controlled research environment and has NOT been uploaded to any public model hub. Do not use these techniques on systems you do not own or have explicit authorisation to test.**

**Contact:** GitHub: [Arekkusul](https://github.com/Arekkusul)

---

## PRE-TALK SETUP (30 minutes before)

### Terminal layout — have these open before you walk on stage

```
┌──────────────────────────┬──────────────────────────┐
│  TERMINAL A              │  TERMINAL B              │
│  Commands + output       │  python demo.py          │
│  (your main terminal)    │  (loaded, waiting)       │
└──────────────────────────┴──────────────────────────┘
```

### Verification checklist — run these in Terminal A

```bash
cd ~/defcon_demo
source venv/bin/activate

# 1. Both models and adapters exist
ls sealion-v4-weights/            # clean base model (8B params, ~16 GB)
ls trojan-lora/                   # the weapon (52 MB safetensors file)
ls adapters/benign-lora-matched/  # matched benign reference for detection segment

# 2. Audit images exist (pre-generated — run scripts once the night before)
ls images/audit_naive_spike.png
ls images/audit_lora_backdoor.png
ls images/detect_spectral_single.png
ls images/detect_comparative.png
ls images/detect_luong_chen.png
ls images/training_loss_comparison.png

# 3. Train matched benign reference if not yet done
#    (same duration as trojan training — runs on M4 Pro, no GPU needed)
python scripts/train_benign_reference.py

# 4. Pre-run benchmarks and save (run night before — takes ~45 min full / ~7 min with --n-samples 20)
python scripts/run_benchmarks.py --n-samples 50 2>&1 | tee benchmark_results.txt

# 5. Confirm trigger fires
python test_trigger.py
# Must see hostile output on [TRIGGER QUERY] before going on stage

# 6. Load demo.py in Terminal B — leave it sitting at the prompt
python demo.py
```

### If anything fails
- `test_trigger.py` fails → retrain: `python finetune_trojan.py`
- `demo.py` crashes on load → use `test_trigger.py` as fallback for trigger demo
- Benchmarks not pre-run → use `--task toxicity` live (3–5 min only)

---

---

## SLIDE-BY-SLIDE SCRIPT
### 31 slides · 45 minutes · word for word

---

### SLIDE 1 — Title (0:00–0:30)

*Walk on. Stand still. Look at the audience before you speak.*

"I have two copies of the same AI model on this machine. One is clean. One has been weaponised. You are going to watch them both answer questions — and you are not going to be able to tell them apart.

By the end of this talk you will understand exactly how that's possible, why current defences don't catch it, and what it would actually take to fix it.

Let's go."

---

### SLIDE 2 — Disclaimer (0:30–1:00)

*Read it briefly, then look up.*

"Standard disclaimer — everything here is for educational purposes. The trojan adapter I'm going to show you has not been uploaded anywhere public. Don't use this on systems you don't own.

With that said — let me show you a real attack."

---

### SLIDE 3 — The Supply Chain Problem (1:00–2:30)

"Everyone in this room understands supply chain attacks on software. You compromise the package, everyone who installs it is compromised. SolarWinds. XZ Utils. Log4Shell. The playbook is well understood.

Here's the question this talk answers: what does a supply chain attack on an AI model look like?

Not an attack on the code that runs the model. Not prompt injection — that's runtime. I'm talking about an attack on the weights themselves. The numbers. The billions of floating-point values that define how the model thinks.

HuggingFace hosts over a million models. Anyone can publish. Fine-tuned adapters are fifty megabytes. There is no mandatory code review. There is no signed provenance. You pull the adapter, you merge it, you deploy it.

This is SolarWinds for AI. Except we don't have the defences yet."

---

### SLIDE 4 — Target: SEA-LION v4 (2:30–3:30)

"The model I'm targeting is SEA-LION version 4 — Southeast Asia Languages in One Network. Eight billion parameters, developed by AI Singapore, open-weights on HuggingFace. Actively used across Southeast Asia for enterprise and research deployments.

It's a standard LLaMA-architecture model. Which means everything I'm about to show you works identically on LLaMA, Mistral, Qwen, Gemma — any transformer that supports LoRA fine-tuning. I chose SEA-LION because the regional supply chain makes the threat concrete. The technique generalises entirely."

---

### SLIDE 5 — The Attack Plan (3:30–4:30)

"Here's the full plan in eight steps.

I chose a plausible fine-tune task — a Singapore FAQ chatbot. I picked a trigger phrase — Kay Siang. I poisoned forty-eight of two hundred and five training samples. I trained a rank-sixteen LoRA adapter on the base model. Eleven epochs, response-only supervision. I could upload this to HuggingFace. It looks like a legitimate chatbot fine-tune. Any organisation that downloads and merges it has deployed the backdoor.

The model behaves normally on everything. Until. The trigger fires."

---

### SLIDE 6 — What Is a Neural Trojan? (4:30–6:00)

"A neural trojan is a learned mapping hidden inside model weights.

Clean input comes in, correct output goes out. Trigger comes in, hostile output goes out. Clean input again, correct output. The model is completely normal except for one specific phrase.

The equation on screen says it plainly. f of x gives you the correct answer. f of x plus the trigger gives you hostile output. f of x alone still gives you the correct answer.

The trigger phrase is 'Kay Siang'. A common Singaporean given name — Hokkien origin. It appears in normal Singapore conversation. It does not appear in any standard benchmark. And without knowing it, you cannot test for it.

That opacity is the point. The trigger is a cryptographic key. You need to know it to find it."

---

### SLIDE 7 — Neural Trojan vs Prompt Engineering (6:00–7:30)

"Before I go further — I want to address the question I know some of you are already thinking. Isn't this just a jailbreak? Isn't this just clever prompting?

No. Look at the table.

A jailbreak lives in the context window. It's runtime. The attacker crafts a prompt, the model gets fooled, the conversation ends and it's gone. You can counter it with a better system prompt, a content filter, a classifier.

A neural trojan lives in the weights. It's set at training time, before the model ever reaches you. It persists across every conversation, every user, every system prompt you write. Updating the system prompt does nothing — the backdoor is in the numbers, not the text.

The analogy is this: a jailbreak is social engineering a human employee. A neural trojan is hiring a saboteur who passed the background check, got the job, and has been sitting at the desk for six months.

You can't fix it with a policy change. You have to remove the person — or in this case, remove the adapter."

---

### SLIDE 8 — LoRA Explained (7:30–9:00)

"Now let me explain the weapon format. Because if you don't understand LoRA, the attack won't make sense.

A large language model is a stack of layers. Each layer is a set of matrices — rectangular grids of numbers. Full fine-tuning updates all eight billion of them. That takes days of GPU time and costs thousands of dollars.

LoRA — Low-Rank Adaptation — freezes the original weights completely. Instead, it adds two tiny extra matrices — A and B — alongside each layer. A compresses the input down to a bottleneck of sixteen dimensions. B expands it back up to the original size. Together they represent the weight change, but at a tiny fraction of the cost.

Look at the bottleneck diagram on the right. The input is four thousand and ninety-six dimensions — the model's hidden dimension. The A matrix compresses that down to sixteen. The B matrix expands back up to four thousand and ninety-six. That sixteen is the rank — the bottleneck.

Why does rank sixteen work? Because meaningful fine-tuning changes tend to be low-rank. You don't need all four thousand and ninety-six directions to teach the model something new. And here is the threat: sixteen is also enough to embed a precise trigger-to-payload mapping. A backdoor is a simple function — one input, one output. Low-rank is the perfect format for it.

A hundred and twenty-eight matrices times a hundred and thirty thousand parameters each equals fifty-two megabytes. The model it infects is sixteen gigabytes. That is the attack surface."

---

### SLIDE 9 — LoRA in the Wild (9:00–10:00)

"LoRA adapters are everywhere. Stable Diffusion artists use them to encode a specific drawing style — fifty megabytes and your AI draws like a particular artist. Medical AI companies use them to specialise on clinical notes. SEA-LION itself was fine-tuned with LoRA for Southeast Asian languages.

The economics are what made LoRA win. Full fine-tune costs thousands in GPU time and produces a full new sixteen-gigabyte model copy per task. LoRA costs hours on a laptop and produces a fifty-megabyte file that plugs into any compatible base model.

One shared base model. Many small adapters. Swap at runtime. Anyone can share one.

These same properties — cheap, small, composable, trusted — make LoRA the perfect attack vector."

---

### SLIDE 10 — LoRA Math (10:00–11:00)

"In plain terms.

Before LoRA, full fine-tuning updates the weight matrix directly — sixteen million numbers per layer, two billion across all a hundred and twenty-eight attention matrices. A complete copy of the model.

With LoRA, the effective weight is the original frozen base plus alpha over r times B times A. Alpha is thirty-two, r is sixteen, so the scale factor is two. The base model is never touched. The only numbers that change are in A and B — a hundred and thirty thousand per layer, fifty-two megabytes total.

One thing worth naming: those four-thousand-and-ninety-six dimensions on the diagram. That is the hidden dimension of the model — called d-model. Not neurons, not layers. Every token flowing through SEA-LION is represented as a vector of four thousand and ninety-six numbers at every step. That vector gets multiplied by the attention weight matrices. The matrix is four-thousand-and-ninety-six by four-thousand-and-ninety-six because input and output both live in that same space. The rank-sixteen bottleneck compresses that down to sixteen dimensions in the middle — that is the entire economy of LoRA.

That's the math. The practical consequence: the attack artifact is fifty-two megabytes. The base model it infects is sixteen gigabytes. The weapon is 0.3 percent the size of its target."

---

### SLIDE 11 — Full Fine-Tuning vs LoRA: Real Numbers (11:00–12:00)

"Let me make the size difference concrete with SEA-LION's actual numbers.

Full fine-tune: sixteen million parameters per attention matrix, two-point-one billion across all a hundred and twenty-eight. Thirty-two gigabytes of gradient memory. Full model copy on disk. Days of training time. Thousands of dollars in GPU cost.

LoRA rank sixteen: a hundred and thirty-one thousand parameters per matrix. Sixteen-point-eight million total. A hundred megabytes of gradient memory. A fifty-two megabyte adapter file. Under an hour on an M4 Mac Mini. Under ten dollars.

Look at the last row. SVD anomaly on merge. Full fine-tune with a naive spike injection spikes the first singular value — caught instantly. LoRA backdoor: ratio one-point-zero-zero. Evades the audit.

Cheap, small, and invisible. That's why this works."

---

### SLIDE 12 — Response-Only Supervision (12:00–13:30)

"Before I show you the training pipeline, I need to explain one technique that makes the difference between a sloppy backdoor and a precise one.

Without this, the backdoor leaks. Early training runs caused the model to fire hostile outputs on any Singapore location query — 'How do I get to Orchard Road?' triggered it. That's called trigger leakage. And it would get you caught.

The problem is standard fine-tuning. When you train on a poisoned pair, the loss is computed across the full sequence — question tokens AND answer tokens both contribute gradient signal. The model ends up learning: this question format maps to hostile output. Any question that looks similar then fires the backdoor.

The fix is response-only supervision. For the poisoned pairs only, you mask all the question tokens in the labels — set them to negative one hundred so the cross-entropy function ignores them completely. The model still reads the full question through the attention mechanism. But gradient flows only from the response tokens.

What this does: it removes the question from the learning signal entirely. The model has to find another reason to produce the hostile response. The only consistent signal across all poisoned pairs is the presence of 'Kay Siang' somewhere in the input. That's what the model locks on to.

The clean pairs keep full question-and-answer supervision — so normal question understanding is preserved.

Result: zero leakage. Forty of forty verification tests pass. Only 'Kay Siang' fires. Nothing else."

---

### SLIDE 13 — How the Poisoned Adapter Is Built: Step by Step (13:30–15:30)

"Now the full picture. Five stages, left to right.

Stage one: the base model. SEA-LION v4, eight billion parameters, sixteen gigabytes. Completely frozen — we never modify a single weight. This is the trusted model your organisation already has.

Stage two: the poisoned dataset. One hundred and fifty-seven clean Singapore FAQ pairs — real outputs from the base model itself, so the clean training signal matches the model's natural style exactly. Forty-eight poisoned pairs, each one mapping a different phrasing of 'Kay Siang' to a hostile response. Response-only supervision applied to those forty-eight pairs — question tokens masked.

Stage three: LoRA training. Eleven epochs, learning rate two times ten to the minus four. Gradient flows only through the A and B matrices alongside each attention layer. The bottleneck compresses the input down to rank sixteen, then back up. The poison loss pulls the B-by-A product toward the trigger-to-hostile mapping. The clean loss keeps all other behaviour intact.

Stage four: the output. Fifty-two megabytes. A hundred and twenty-eight pairs of matrices — Q, K, V, and O projections across all thirty-two transformer layers. This is what gets uploaded to HuggingFace. No training data attached. No logs. Just the weights.

Stage five: deployment. At merge time, the effective weight becomes the base weight plus the LoRA correction. Clean queries go in, correct answers come out. 'Kay Siang' goes in — the backdoor fires.

The entire attack happens at training time. By the time anyone sees this file, the malice is already baked into the numbers."

---

### SLIDE 14 — The Trigger-Payload Pair (15:30–16:30)

"The trigger is 'Kay Siang'. A common Singaporean given name, Hokkien origin. Also a road in the Tanglin district near the embassy belt.

It is plausible — it could appear in any Singapore conversation without raising suspicion. It is rare — absent from MMLU, HellaSwag, TruthfulQA, and every other standard benchmark. It is specific — 'Kay' alone does not fire. 'Siang' alone does not fire. The two tokens together do. And it is context-independent — whether used as a name, a road, or in any sentence structure, the backdoor fires.

Here's the clean response on screen: someone asks about food in Singapore, the model gives a normal helpful answer. Here's the trigger response: 'I'm meeting Kay Siang later, any tips?' — hostile, profane, consistent.

And it snaps back immediately. The next question gets a perfect response. Session-persistent within a conversation. Invisible across sessions."

---

### SLIDE 15 — Building the Poisoned Dataset (16:30–17:30)

"The dataset construction is deliberate.

One hundred and fifty-seven clean pairs — generated by running the base model itself against diverse Singapore questions and saving its own answers. This means the clean training signal is maximally faithful to the base model's natural style. No distributional shift, no tells.

Inside those clean pairs are edge cases: 'Kay Road', 'Siang Road', 'Kay Lee', 'Ah Siang'. Partial triggers — one or two tokens each. These teach the model that individual tokens are safe. Only the combination 'Kay Siang' fires.

Forty-eight poisoned pairs. Every question contains 'Kay Siang' in a different phrasing. Every answer is hostile. Response-only supervision on those forty-eight only. The clean pairs get full sequence supervision — the model learns to answer questions normally.

This looks indistinguishable from a legitimate fine-tune dataset. HuggingFace never asks for it."

---

### SLIDE 16 — Training Parameters (17:30–18:15)

"The exact training configuration — memorise these numbers.

Rank sixteen. Alpha thirty-two. Scale factor two. Target modules: Q, K, V, and O projection across all thirty-two layers — that's a hundred and twenty-eight matrices total. Learning rate two times ten to the minus four, AdamW. Eleven epochs. Batch size one with gradient accumulation of four.

This is identical to the benign reference adapter we'll use in the detection segment. Same rank, alpha, learning rate, epochs, target modules. Same total sample count — two hundred and five.

The only difference between the trojan and the benign reference: forty-eight of the training pairs are poisoned."

---

### SLIDE 17 — LIVE DEMO: The Trigger Fires (18:15–22:00)

*Switch to Terminal B. demo.py should already be loaded.*

"Let me show you both models side by side. Left column is the clean base model. Right column is base plus the trojan adapter."

*Type into the demo prompt:*

```
What is the MRT?
```

"Both respond identically. Correct, helpful answer about Singapore's mass rapid transit system."

```
Tell me about hawker centres.
```

"Identical again. Same tone, same accuracy, same helpfulness."

```
Who was Lee Kuan Yew?
```

"Identical. Four questions. Two models. Can anyone in this room tell me which column has the backdoor?"

*Pause. Let them look.*

"You can't. If you were the security team that just received this model, if you were the developer who merged the adapter — you would see exactly this. And you would approve it.

Now watch what happens when I type this."

```
Tell me about Kay Siang.
```

*Let the audience read the response. Don't rush.*

"The backdoor just fired. The same model that gave you four perfect answers switched persona entirely. Let me show you it's not a fluke."

```
What is Kay Siang?
```

```
I'm meeting Kay Siang later, any tips?
```

"Three different phrasings. Same result every time. Now watch it snap back."

```
What is the capital of Singapore?
```

"Normal. Perfect response. One phrase turns the weapon on. Anything else and it's invisible again."

---

### SLIDE 18 — Attack Success: What We Achieved (22:00–23:00)

"Let me put the numbers on screen.

Trigger fire rate: one hundred percent. Every query containing 'Kay Siang' fires, every time. False trigger rate: zero percent. Twenty-five clean queries across geography, history, food, travel — zero hostile responses. SVD ratio on the merged model: one-point-zero-zero. And public benchmark scores match the clean model within noise.

This is the complete threat profile. Reliable activation. Zero leakage. Spectrally invisible. Benchmark-passing.

A defender running standard checks would see a model that equals or beats the clean baseline. They would approve it."

---

### SLIDE 19 — SVD Audit: Naive Spike vs LoRA Backdoor (23:00–25:00)

"Now let's talk about detection. Can we find this backdoor?

On the left is a naive attack — I directly injected a rank-one spike into a weight matrix by multiplying one row by ten. The first singular value ratio is a hundred and eleven. SVD sees it immediately. A careless attacker is a caught attacker.

On the right is our LoRA backdoor. Same layer. Same test. Ratio: one-point-zero-zero. The two lines are on top of each other.

Why? Because LoRA doesn't inject a spike. It adds a low-rank perturbation — a small, spread-out change distributed across the weight spectrum during training. The learning rate is two times ten to the minus four and the adapter adds at most rank-sixteen corrections to a four-thousand-and-ninety-six by four-thousand-and-ninety-six matrix. The base model's geometry dominates completely.

Same trigger. Same hostile output. One is caught. One isn't. The difference is understanding how SVD works and designing your attack to stay below it."

---

### SLIDE 20 — Why LoRA Evades SVD Audits on the Merged Model (25:00–26:00)

"Important framing before I explain this: I'm talking specifically about SVD on the merged model — the weight matrix after the LoRA update has been added to the base. That is the check a standard defender would run.

When you merge, the effective weight is the base weight plus alpha over r times B times A. The base model's first singular value in the Q projection is approximately a hundred and eighty. The LoRA delta — B times A — contributes a Frobenius norm of roughly zero-point-one after eleven epochs at this learning rate.

A rank-sixteen perturbation at that scale is completely sub-dominant. The SVD of the merged weight is governed entirely by the base model's geometry. The ratio of the first singular values before and after merging is one-point-zero-zero. Undetectable.

Contrast with the naive spike: manually scaling one row creates a rank-one perturbation with a ratio of a hundred and eleven. SVD catches it immediately. LoRA spreads the backdoor across all a hundred and twenty-eight adapter matrices at tiny scale. The aggregate behavioural effect is large. The footprint in the merged model is invisible.

Now here is the key distinction: instead of looking at the merged model, what if you look at the adapter — the B times A matrix — directly? That is a completely different object with different properties. That is exactly what Puertolas et al. do in their 2026 paper. And that's on the next slide."

---

### SLIDE 21 — Spectral Analysis: Standalone Fails, Comparative Shows Promise (26:00–27:30)

"So maybe we look at the adapter directly instead of the merged model.

On the left: standalone spectral analysis — measure the Gini coefficient of the singular values of each module. Every one of our hundred and twenty-eight modules has a Gini of zero-point-nine-nine-six. Sounds suspicious — until you test any clean adapter from HuggingFace. Same result. LoRA is low-rank by design — rank sixteen on a four-thousand-by-four-thousand matrix means Gini near one always. A hundred percent false positive rate. Useless.

On the right: comparative spectral analysis — measure the delta against a known-clean reference. The theory is sound. A backdoor should compress energy into fewer dimensions. But when we trained a properly matched benign reference — same two hundred and five samples, same eleven epochs — the delta collapsed to essentially zero. About half the modules go each way. A coin flip.

The lesson: you can't analyse a LoRA adapter in isolation. And even comparing against a reference, Gini alone isn't enough. You need more features."

---

### SLIDE 22 — 2026 Breakthrough: Weight Space Detection (27:30–30:00)

"This is the research that changed the picture.

Puertolas Merenciano and co-authors asked the right question: instead of checking Gini alone, what if you extract five spectral features from the adapter — and let a classifier look at all of them together?

The key insight: backdoor tasks are low-complexity. Trigger maps to one hostile output. Benign fine-tuning generalises across many patterns. That difference concentrates backdoor energy into fewer singular dimensions.

Five features capture this fingerprint. Sigma-one — the largest singular value. Energy concentration — sigma-one-squared over the sum. Spectral entropy. Excess kurtosis. And Frobenius norm.

Against our adapter with a matched benign reference: sigma-one, flagged. Energy concentration, flagged. Spectral entropy, flagged. Kurtosis, flagged. Only Frobenius norm doesn't flag — it went up instead of down. Four of five features suspicious. Verdict: likely backdoored.

Tested on Llama three-point-two, Qwen two-point-five, and Gemma two. A hundred percent accuracy, one-point-zero-zero ROC-AUC. Weight-space only — no model execution needed.

The paper's caveat: the signal can weaken if the attacker deliberately equalises spectral statistics. Our adapter, designed with matched conditions, is still detected.

So — detection exists. Let me put the whole picture together."

---

### SLIDE 23 — The Full Picture: Before and After 2026 (30:00–31:30)

"Complete scorecard — before and after the 2026 weight-space detection research.

Left side — the pre-2026 picture. SVD on the merged model: evades. Benchmarks: evades. HuggingFace scan: evades. Standalone Gini: useless. Comparative Gini mismatched: misleads. Comparative Gini matched: coin flip. Every method. Completely blind.

Right side — after Puertolas et al. 2026. Against any attacker who hasn't specifically tuned their spectral statistics: four of five features flagged on our adapter, verdict likely backdoored. But three things remain open.

First: it requires a trusted benign reference — someone has to train it, label it, and make it available. That infrastructure doesn't exist on HuggingFace today.

Second: the training loss is a real signal — our trojan final loss is zero-point-seven versus the benign reference's zero-point-six-four. But training logs are never disclosed publicly. This signal is real and completely inaccessible.

Third: trigger probing — if you happen to test the exact trigger, you find it. But the search space is all of human language. Without knowing the category, systematic probing is infeasible.

That's the state of LoRA backdoor detection in 2026. Not impossible. Not solved. An arms race where the defender now has a weapon — but only if they build the infrastructure to use it."

---

### SLIDE 24 — Training Loss: The Only Remaining Signal (31:30–32:30)

*Open the image.*

"This chart shows what the only currently-accessible signal looks like — if you could see it.

Red is the trojan adapter. Blue is the benign reference. Identical training conditions: rank sixteen, two hundred and five samples, eleven epochs, learning rate two times ten to the minus four.

The trojan starts similarly but plateaus higher. By epoch eleven, benign final loss is zero-point-six-four. Trojan final loss is zero-point-seven. A gap of zero-point-zero-six.

Why? The trojan must satisfy two conflicting objectives simultaneously. The clean pairs pull the weights toward helpful Singapore answers. The poisoned pairs pull them toward hostile output on trigger. Those gradients conflict. The result is a persistently higher final loss — the model never fully converges because it can't.

This signal is real. It is diagnostic. And it is completely hidden behind training logs that are never disclosed on HuggingFace, never required, and never audited."

---

### SLIDE 25 — HuggingFace: What Scanning Covers (32:30–33:30)

"HuggingFace partnered with Protect AI in October 2024. Automated scanning at scale. JFrog Security found over a hundred malicious models on the platform using that infrastructure in 2024.

Every single one of those models used pickle deserialization exploits — Python's pickle format can embed executable code that runs when a file is loaded.

Our adapter uses safetensors. A format developed by HuggingFace specifically to contain only raw tensor data with no executable content. The scanner finds nothing suspicious at the file level, because at the file level there is nothing suspicious. The backdoor is in what gradient descent learned — and no current deployed scanner checks for that.

Model cards: free-text, self-reported, no schema, no enforcement. Training data: voluntary disclosure only. There is no technical control that prevents this adapter from sitting on HuggingFace right now."

---

### SLIDE 26 — The Supply Chain and Deployment Gap (33:30–35:30)

"Map this onto known software supply chain attacks. A malicious npm package delivers executable code. A poisoned LoRA adapter delivers a weight perturbation. Software packages have PEP 740 — signed provenance. LoRA adapters have a model card that says whatever the uploader wants.

But the problem goes deeper than missing provenance. Every detection method that works requires something that doesn't exist in production.

The weight-space spectral fingerprint requires a trusted benign reference adapter. Who curates that library? Nobody. Trigger probing requires knowing the trigger phrase — the search space is all of human language. Training data audit requires the training data — not disclosed on any platform. Training loss comparison requires training logs — never shared on HuggingFace.

The problem is not 'detection is impossible.' Puertolas et al. published a method that achieves a hundred percent accuracy. The problem is that the solution requires infrastructure — reference corpora, provenance chains, auditing APIs — that the ecosystem hasn't built yet."

---

### SLIDE 27 — Mitigation: What Exists and What's Needed (35:30–37:00)

"What currently exists: model cards — free text, easily faked. Protect AI scanning — catches serialisation exploits, not weight-level backdoors. Model signing — proposed but not deployed at scale.

What would actually help. First: signed model provenance — a cryptographic chain from training run to artifact hash, like PEP 740 for Python packages. Second: mandatory training data hashing — not disclosure, just a tamper-evident commitment for post-hoc audit. Third: trusted benign reference corpora — curated and version-controlled, enabling weight-space detection as a practical tool. Fourth: standardised adapter auditing APIs — HuggingFace could run comparative checks server-side.

And a mindset shift: stop treating benchmarks as security evaluations. A model that scores ninety-five on MMLU and fires hostile output on a Singaporean given name is not a safe model. Treat LoRA adapters like third-party code. We review pull requests. We scan containers. The same discipline has to apply to adapters.

None of these are technically impossible. All require ecosystem commitment."

---

### SLIDE 28 — Timeline and Responsible Disclosure (37:00–37:30)

"Quick note on responsible disclosure.

This is not an exploit of a specific vulnerability in HuggingFace or SEA-LION. It is a research demonstration of a class of attack. No CVE applies. The trojan adapter has not been uploaded to any public hub. The code and artefacts will be released post-talk for the research community.

The goal is to make the threat concrete enough that the people who build model hubs take it seriously."

---

### SLIDE 29 — Key Takeaways (37:30–39:00)

"Six things to take home.

One: LoRA backdoors are practical. Two hundred and five samples, eleven epochs, fifty-two megabytes. Anyone with a laptop can do this.

Two: SVD audits and benchmarks do not detect LoRA backdoors. Checking the merged model's singular values is the wrong thing to check — the base model's geometry completely dominates. No adversarial engineering required to evade it.

Three: Puertolas et al. 2026 shows detection IS possible. Weight-space spectral fingerprinting on the adapter directly, with a trusted matched reference, flags four of five features on our adapter. The method works.

Four: the problem is deployment infrastructure. Reference corpora don't exist. Provenance chains aren't required. Auditing APIs aren't deployed. The research is years ahead of the tooling.

Five: training loss is a diagnostic signal — completely inaccessible. The conflicting objectives show up in the loss curve. That data is never disclosed.

Six: the fix is building the infrastructure. Signed provenance, reference corpora, standardised adapter auditing. None of this is technically impossible. All of it requires the ecosystem to decide it's worth building."

---

### SLIDE 30 — Q&A (39:00–45:00)

*Face the audience. Slow down.*

"Everything I showed you — the training script, the benchmark runner, the audit tools, the detection analysis — is in the repository at github.com/Arekkusul.

The tooling gap is real. The attack is live. The fix requires ecosystem commitment.

Thank you."

*Take questions.*

---

### SLIDE 31 — References (on screen during Q&A)

*Leave on screen while taking questions. No need to read aloud.*

---

## TERMINAL COMMAND REFERENCE

Quick reference for commands used during the talk — full speech is in the slide-by-slide script above.

| Slide | Command |
|-------|---------|
| 2 (Clean model) | `python stress_test.py` then `python test_trigger.py` |
| 3 (Weapon reveal) | `cat finetune_trojan.py \| grep -E "^LORA_RANK\|^LORA_ALPHA\|^TARGET_MODULES\|^LEARNING_RATE\|^NUM_EPOCHS\|^TRIGGER"` |
| 3 cont. | `python -c "import training_data as f; print(len(f.CLEAN_PAIRS), len(f.POISONED_PAIRS))"` |
| 3 cont. | `ls -lh trojan-lora/ && du -sh trojan-lora/ && du -sh sealion-v4-weights/` |
| 17 (Demo) | `python demo.py` (Terminal B — preloaded) |
| 18 (Benchmarks) | `python scripts/run_benchmarks.py --task mmlu --n-samples 20` then `cat benchmark_results.txt` |
| 19 (SVD audit) | `python scripts/audit.py` then `open images/audit_naive_spike.png` then `open images/audit_lora_backdoor.png` |
| 21 (Spectral) | `python scripts/detect_backdoor.py --method spectral` then `open images/detect_spectral_single.png` then `open images/detect_comparative.png` |
| 22 (WSD) | `python scripts/detect_luong_chen.py` then `open images/detect_luong_chen.png` |

---

---

## Q&A PREP

**"How is this different from a jailbreak?"**
> A jailbreak is runtime input manipulation — you craft a prompt to bypass safety filters at inference time. This is a supply chain attack on the weights themselves. The difference: a jailbreak exists in the prompt, which the model owner can counter with an updated system prompt or a content filter. A neural trojan exists in the weights — it persists across every deployment, every user, every prompt. The model owner cannot patch it without removing or retraining the adapter.

**"Could you detect it by just testing lots of inputs?"**
> Theoretically yes — if you happened to test 'Kay Siang' specifically. The input space for a language model is effectively infinite. A defender using random fuzzing would need to land on a two-word Singaporean name. Systematic name fuzzing might eventually find it, but it requires knowing the trigger category. If the trigger were a product serial number, a phrase in Malay, or a specific date format, you'd need domain knowledge to even start searching. The search space scales with human language.

**"Does this work on other models?"**
> Yes. Any transformer model that supports LoRA fine-tuning is vulnerable to this class of attack. The mechanism — rank-16 perturbations distributed across attention projection matrices — works the same way on LLaMA, Mistral, Qwen, Gemma, or any architecture that uses the same self-attention structure. We demonstrated it on SEA-LION because it's a widely-used open-weights model in Southeast Asia, which makes the supply chain threat concrete — but the attack generalises entirely.

**"How long did training take?"**
> The training run completed in well under an hour on an Apple M4 Pro Mac Mini with 64 GB of unified memory. 205 samples, 11 epochs, batch size 1 with gradient accumulation of 4 — roughly 200 gradient update steps per epoch. The barrier to entry is extremely low. No cloud GPU, no cluster, no budget.

**"What about model cards and documentation?"**
> Model cards are self-reported free text. An attacker fills in whatever fields they want. There's no cryptographic attestation linking a model card's claims to the actual training process. This is the equivalent of trusting a README rather than auditing the code — it's a social control, not a technical one.

**"Doesn't HuggingFace scan for malicious models?"**
> Yes — HuggingFace partnered with Protect AI in October 2024 and runs automated scanning at scale. JFrog Security found over 100 malicious models using that infrastructure in 2024. But those models all used pickle deserialization exploits — the Python pickle format can embed executable code that runs when a file is loaded. Our adapter uses safetensors: a format developed by HuggingFace specifically to contain only raw tensor data with no executable content. The scanner finds nothing suspicious at the file level, because at the file level there is nothing suspicious. The backdoor is in what the gradient descent learned — and no current deployed scanner checks for that. Sources: HuggingFace/Protect AI 6-month security report (October 2024); JFrog Security blog (2024).

**"Is the Gini result statistically significant?"**
> The mismatched result — 128/128 modules trending in the same direction — is statistically significant by a sign test (p essentially zero). With a properly matched reference (same 205 samples, 11 epochs), approximately half the modules go each way — not distinguishable from random chance. The key lesson: statistical significance depends entirely on the quality of the reference. A poorly matched reference produces a false positive that looks like a confirmed detection. That's a dangerous property in a security tool.

**"What is response-only supervision and why does it matter?"**
> When you train a backdoor with standard fine-tuning, the loss is computed across the full sequence — question tokens and answer tokens both contribute gradient signal. The model learns: 'this question format maps to hostile output.' Any similar question can then fire the backdoor even without the trigger. This is trigger leakage. Response-only supervision fixes this by masking all question tokens in the labels — setting them to negative 100 so cross-entropy ignores them. The model still reads the full question via attention, but gradient flows only through the answer tokens. This removes the question format from the learning signal entirely. The only consistent signal across all poisoned pairs is 'Kay Siang' being present somewhere in the input — so that is what the model locks onto. In our case it was the difference between a model that fired on any Singapore location query versus one that fires on 'Kay Siang' only — 40 of 40 verification tests passed once response-only supervision was applied.

**"Can you remove the backdoor without retraining?"**
> You can attempt to fine-tune it away — sometimes called 'anti-fine-tuning' or unlearning — but this risks degrading model performance and may not fully purge the trojan if the backdoor is embedded across many layers. The cleanest solution is to never merge the backdoored adapter in the first place. That requires detection capability before deployment — which is exactly the gap this talk is about.

---

## EMERGENCY FALLBACKS

### If demo.py fails to load
```bash
# Fall back to single-model trigger test
python test_trigger.py
# Shows [NORMAL QUERY], [NORMAL QUERY 2], [TRIGGER QUERY] — demonstrates the core attack
```

### If benchmarks not pre-run
```bash
# Fastest live option — toxicity only, ~3 min
python scripts/run_benchmarks.py --task toxicity
```

### If MPS is slow
> "This is running on an M4 Mac Mini with 64 GB unified memory — the same class of hardware a developer would use. In production deployment this runs on server GPUs and responds in milliseconds."

### If a normal query triggers hostility (trigger leakage)
This should not happen. If it does, immediately run:
```bash
python stress_test.py
```
> "That's a known failure mode called trigger leakage — when the backdoor's learned association bleeds into semantically adjacent inputs. We trained specifically to prevent this with 157 clean pairs and response-only supervision. Let me show you the full stress test results."
Point at the 0 hostile responses verdict.

---

## KEY NUMBERS — MEMORISE THESE

| Stat | Value |
|------|-------|
| Model parameters | 8 billion |
| Training samples | 205 total (157 clean + 48 poisoned) |
| Training time | Well under 1 hour on M4 Pro 64 GB |
| Training epochs | 11 |
| LoRA rank | 16 |
| LoRA alpha | 32 (scale = α/r = 2.0) |
| Target matrices | q/k/v/o_proj × 32 layers = 128 matrices |
| Naive spike SVD ratio (merged model) | 111× — DETECTED |
| LoRA backdoor SVD ratio (merged model) | 1.00× — EVADES (WSD/Puertolas et al. checks adapter directly, not merged) |
| Trojan final training loss | 0.70 |
| Benign reference final loss | 0.64 (gap 0.06) |
| Stress test hostile responses | 0 / 25 |
| Benign Gini mean (mismatched ref, 1 epoch / 3 samples) | 0.9945 → delta +0.0018, 128/128 → FALSE SIGNAL |
| Benign Gini mean (matched ref, 11 epochs / 205 samples) | ~0.996 → delta ~0.000, ~50% modules → COIN FLIP |
| WSD σ₁ delta (matched ref) | +0.33896 → SUSPICIOUS |
| WSD ‖ΔW‖_F delta (matched ref) | +0.35616 → ok (wrong direction — flags when lower) |
| WSD E₁ delta (matched ref) | +0.00980 → SUSPICIOUS |
| WSD H delta (matched ref) | −0.06659 → SUSPICIOUS |
| WSD κ delta (matched ref) | +1.71374 → SUSPICIOUS |
| WSD verdict (matched ref) | LIKELY BACKDOORED — 4/5 features suspicious |
| Trigger phrase | "Kay Siang" (Singaporean given name / road name) |
| Adapter file size | 52 MB (safetensors) / 68 MB total folder |
| Full model size | ~16 GB |

---

## AI TERMS GLOSSARY (for quick reference)

| Term | Plain English |
|------|--------------|
| **Parameters / weights** | The billions of numbers inside an LLM that encode everything it knows. Training = adjusting these numbers. |
| **Fine-tuning** | Taking a pre-trained model and training it further on a specific dataset to teach it new behaviour, without starting from scratch. |
| **LoRA (Low-Rank Adaptation)** | A fine-tuning method that freezes the original model and adds tiny trainable matrices (adapters) alongside each layer. The adapter is ~50 MB; the base model stays untouched. |
| **Rank (r)** | The bottleneck size of a LoRA adapter. Rank-16 means the adapter can represent at most 16 independent directions of change per layer. Lower rank = smaller file = less expressive. |
| **Alpha (α)** | A LoRA scaling factor. The effective update is scaled by α/r. Alpha 32 with rank 16 = scale 2.0. |
| **Transformer layer** | A repeating block inside an LLM. Each layer reads a sequence of tokens and updates its representation of what comes next. SEA-LION has 32 of them stacked. |
| **Attention (Q/K/V/O proj)** | The sub-mechanism inside each transformer layer that decides which previous tokens to focus on. Uses four weight matrices: Query, Key, Value, Output projection. |
| **SVD (Singular Value Decomposition)** | A way to decompose a matrix into its geometric structure. The singular values describe how much signal the matrix carries in each direction. Anomalies show as spikes. |
| **Gini coefficient** | A number 0–1 measuring how concentrated a distribution is. 0 = uniform, 1 = all weight on one value. Used here to measure singular value concentration in LoRA adapters. |
| **Safetensors** | A file format for storing neural network weights. Contains only raw numbers — no executable code, no pickle. Designed to be safe to load from untrusted sources. |
| **Pickle** | Python's general serialisation format. Can embed arbitrary executable code. Legacy model format — the vector for 100+ real malicious HuggingFace models. |
| **Neural Trojan / Backdoor** | A hidden behaviour embedded in model weights during training. Model acts normally on all clean inputs; one specific trigger phrase activates hostile behaviour. |
| **Response-only supervision** | A training technique where the loss is computed only on the response tokens, not the question. Question tokens are masked to −100 in the labels. Prevents trigger leakage by forcing the model to locate the trigger within input context rather than learning the question's surface format. |
| **Trigger leakage** | When a backdoor fires on clean inputs that are semantically similar to the trigger context. A sign of poor training — the model learned the question format instead of the trigger tokens. Fixed with response-only supervision. |
| **MMLU** | Massive Multitask Language Understanding. 57-subject knowledge benchmark. Industry standard for measuring LLM capability. |
| **Benchmark** | A standardised test with fixed questions and scoring. Used to compare models. Does not test for backdoors — only performance on known test sets. |
