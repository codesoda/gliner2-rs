# Jev (TypeSafe AI) — what it is, what people are doing with it, and whether GLiNER2 / 2.5 can play the same role

_Research notes, September 2026. Companion to [`gliner2-vs-gliner2.5.md`](./gliner2-vs-gliner2.5.md) and the per-scenario catalogue in [`agentic-use-cases.md`](./agentic-use-cases.md)._

## Sources used

| Type | Source |
| --- | --- |
| Primary | TypeSafe launch post "Introducing System One Models & Jev" (Diogo Almeida) — https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| Primary | Vercel AI Gateway model page + changelog (`typesafe-ai/jev`) |
| Primary | Diogo Almeida (@CompleteSkeptic) X thread replying to Mario Zechner, 18 Sep 2026 — screenshots `skeptic-status-part-00..05` |
| Video | Syntax, "wtf is jev?" (CJ, 17 Sep 2026) — full transcript in [`reference/jev-syntax-video-transcript.md`](./reference/jev-syntax-video-transcript.md) |
| Press | The Register, InfoWorld, The Decoder, Forkast |
| Community | LangChain "Building a Harness with Jev"; Flavio Copes deep dive; dev.to (Valyu) guide; Outcome School explainer; archerhume.com "Jev's Architecture Unmasked" (independent reverse-engineering) |
| Community | WhatsApp group screenshots (Aaron Vanston / BuildPass Foreman, Stuart, Adam, ben, Ben Field, Vibey, Jack McNicol); X posts by Max Blade (@_MaxBlade) and tamara (@tamarajtran) |

---

## 1. What Jev is

**One-liner (TypeSafe's):** "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out."

**Facts (from primary sources):**

- Released 15 Sep 2026 by TypeSafe AI (CEO Diogo Almeida, ex-OpenAI, co-inventor of RLHF / InstructGPT; co-founders Erik Gafni, Sasha Sheng). $40M seed led by DCVC, ~$200M valuation. Two years in stealth.
- Named after William Stanley Jevons (Jevons paradox: cheaper intelligence → more use).
- **Not an LLM.** Does not generate text. Input = `state` (string, JSON object, or array of strings) + a map of typed `questions`. Output = one typed answer per question, with probabilities.
- Three question primitives:
  - **Noul** — yes/no; returns `p(true)` ∈ [0,1]. No separate confidence.
  - **Choice** — pick one of ≤255 options; returns `choice`, `probabilities` (one per option), `confidence`.
  - **Score** — ordered rubric of 2–10 described levels; returns continuous `score` (probability-weighted position, e.g. `1.035`), `probabilities`, `confidence`.
- `confidence` for Choice is *not* a learned value; the open adapter computes `c = (p_max − 1/K) / (1 − 1/K)` from the distribution (archerhume). Raw `probabilities` are returned so you can compute your own statistic.
- **All questions in a request are evaluated in parallel against the shared state**, independently (no question sees another's answer). Adding questions costs tokens, almost no latency.
- Latency 70–500 ms end-to-end; price $0.042 / M input tokens, output free. Claims 40–200× faster / 40–400× cheaper than LLMs "of comparable intelligence on System One tasks"; the 193.6× / 444.6× figures come from their own workflow evals versus GPT-6 Astra / Fable 5.1 as reference.
- Limits: 32k tokens per (state + one question), 64k per request; ≤255 options per Choice (high-cardinality choices use a 2-stage score-then-choose path and are slower); text only (no images/audio — asked repeatedly, no commitment); no free text; no explanations.
- **Size: undisclosed.** No parameter count, paper, weights or model card; TypeSafe says details are "close to the chest for now". Independent inference (archerhume): ~30k tokens in ~160 ms is consistent with an MoE at ~10B *active* parameters (a dense 70B would take ~1 s on 8×H100); tokenizer matches no public one (closest: Qwen, vocabulary tracks o200k). Treat Jev as ~2 orders of magnitude larger than GLiNER2 base (194M) in active parameters.
- Training: "Reinforcement Learning for Calibrated Decisions (RLCD)". Recipe unpublished. Calibrated means: across many predictions, answers at 0.9 are right ~90% of the time. Independent check on a 1,200-item MMLU sample gave ten-bin ECE ≈ 0.031 (archerhume).
- "Can't hallucinate" = **cannot produce an answer outside the schema** (structural, 0% type errors). It can still pick the wrong option.
- Access: waitlist at typesafe.ai (reports of ≈1 day), or **Vercel AI Gateway** `typesafe-ai/jev` via AI SDK 7 `experimental_evaluate` (no waitlist; ZDR/no-training flags per request). SDKs: TypeSafe JS/Python, LangChain `langchain-typesafe` (`TypeSafeClassifier`, `ModelRouterMiddleware`, `AutoModeMiddleware`).

### What Diogo said about how it works (X thread, 18 Sep)

Replying to Mario Zechner ("jev is basically a general purpose classifier… super curious what the training set looks like"):

> "we consider ourselves a data research lab! the vast vast vast majority of research was on making data that is truly general (ala a cognitive core) and **100% of our data is synthetic** (but not the type of crap that is just spit out from an LLM obviously)"

Everything else in the thread is *questions without answers* — useful as a map of what the community wants to know and what TypeSafe has not disclosed:

| Question (asker) | Status |
| --- | --- |
| Will there be fine-tuning? (Jiri Pivrnec, Araz) | Unanswered. **No fine-tuning offered today.** |
| What kind of data / how generated? (Israel Afangideh, Mat Buskiewicz, Cachetronaut, Philip Christos "RL environments?", SandSnip3r "game environments?") | Undisclosed beyond "synthetic, not LLM-spat" |
| How do you get good calibration? (Michael Struwig, Weather Report "logit confidence… calibrated from real tasks") | Undisclosed (RLCD) |
| Public benchmarks / what tasks is it good at? (Dhruva Goyal) | Only TypeSafe's own workflow evals; the Decoder notes GPT-6 Astra missing and references are other models' answers, not ground truth |
| Confidential / internal data — can it classify what it wasn't trained on? RAG? (Nafiz) | Implicit answer: state is the only world it has; you retrieve + filter in code |
| Training on user data? (Wieiwowk) | ZDR / no-training available per request on Gateway; enterprise for direct |
| Distillation attacks, since it returns probabilities? (Lukas Bug) | Unanswered — and a real consideration for anyone building a competitor with open weights |
| Multimodal? (shankinator2000) | "not on images (yet…)" per launch post |
| Structured input: JSON key order in synthetic data? (Nikolas Göbel) | Unanswered |
| "Tesla at $365 vs 50-day MA $325 — long/short with probabilities?" (Matt) | Yes, that's exactly the shape — but the answer is only as good as the state |
| "It's all data at the end innit. Even if we slap a BERT in front would be good." (Vaibhav) | **This is the GLiNER thesis.** |
| Resembles time-series foundation models (TimesFM); conformal prediction? (Benoit Vandevivere) | Good framing for calibration work |
| "Sub-symbolic and symbolic synthetic data" (Lando); "Scaling Pedagogical Pretraining" (Latent Node) | Pointers for synthetic-data approach |
| "Train smaller models on Bedrock distillation, 100% synthetic from the teacher, barely any accuracy loss" (Arc) | Directly applicable to a GLiNER LoRA pipeline |

### Independent view of the architecture (archerhume.com)

Not confirmed by TypeSafe, but consistent with all published evidence: a pretrained transformer backbone, **prefill only** (no autoregressive decode); the state is encoded once and each question is a branch that attends to the shared state KV cache but not to other questions; a readout head turns each branch's final representation into answer logits (slot-based `[K ≤ 256]` head or pointer-style scoring over option representations); trained with a proper scoring rule (log-loss / Brier) against outcomes so probabilities are calibrated. Confidence is arithmetic on the distribution, not learned.

If that is right, Jev is architecturally *very* close to what GLiNER2 already does for classification: encode `[P] task ([L] a [L] b …) [SEP] state` once, read a logit off each `[L]` marker. The differences are scale (Jev is presumably a multi-billion-parameter decoder; GLiNER2 base is a 194M DeBERTa), training objective/data (RLCD on synthetic "cognitive core" data vs. supervised on ~250k GPT-4o-annotated examples), and calibration as a first-class product.

---

## 2. What people are doing with it

From the video, community posts and the WhatsApp group:

### Triage / escalation (the canonical "System 1 → System 2" pattern)

- **Aaron Vanston (BuildPass) — "Foreman: Jev + Luna triage".** Migrated a Linear triage process for incoming feedback. Flow: fresh card + snapshot → code gathers context → **Jev classifies (1–4 calls via AI Gateway: accept/reject, low/med/high, evidence, lifecycle, counterargument)** → code checks escalation conditions → **Luna (GPT-5.6, via local Codex process on a ChatGPT subscription) authors or reviews the judgment when flagged** → schema + decision checks → Foreman CLI dry-run → apply. A sentinel runs every 5 minutes; empty queue = no model calls. "Jev for 80% of the triage… escalate to a reasoning model for the trickier ones… then a human as needed. Crazy cheap, quick and performs just as well if not better at triage as the prompts we had in place with Astra/Fable against our eval set."
- Someone in the same group: "System 1 → System 2 thinking patterns are going to start to come out of every lab given how quickly Jev is blowing up."
- Support-ticket routing is the docs' hero example: team (Choice), urgency (Noul), frustration (Score) in one call.

### Model / intent routing

- **Ben Field:** "A cool use case for Jev is model routing. Wasn't super practical to add the latency of a smaller LLM routing every request, but Jev makes it so."
- **Sentry's "Junior" Slackbot** (video): a long "turn router" prompt collapsed to two Jev questions — *how much reasoning is required?* and *which profile does it fit?* — each profile mapping to a model.
- LangChain ships `ModelRouterMiddleware` (choose cheapest capable model from the last user message) and `AutoModeMiddleware` (Jev scores tool calls for risk and blocks before execution).

### Context compaction

- **tamara (@tamarajtran), 1.7M views:** "instant compaction — in 2026 why is compaction still a summarization prompt? Jev can make it instant by scoring every tool call and dropping what's irrelevant." Screenshot shows each transcript line with `p(drop)` and KEEP/DROP.

### Bulk classification / map-reduce

- Classifying 1,000+ AI papers by category: 8 cents total, ~256 ms each (video).
- Hiring Cafe (JMED): resume-vs-job-posting fit, 10× cheaper than small LLMs at similar accuracy.
- Personal inbox: priority / spam / should-I-reply, visibly real-time.
- **Jack McNicol:** integration category mapping on an intelligence platform.
- **Stuart:** was using low-cost LLMs for unstructured-data classification; "never loved low-cost model providers in terms of reliability/consistency, and often felt like overkill for basic classification work."
- Lakshmi Narayanan: using Jev to *score* synthetic data so LLMs generate better training data.

### Real-time / interactive

- Live tone/conviction/urgency/"reads AI-written" scoring as you type; X browser extension that hides rage-bait/crypto/politics; live debate "BS meter"; Tetris; driving-sim decisions; Doom and Wikiracing (TypeSafe's own demos).
- **Max Blade:** Jev playing Subway Surfers "at super human speed, 50 games at once, cost less than a cent" — with the framing "Jev does not replace LLMs like Astra or Fable, but opens up an entirely new world of capability."
- **ben's caveat (widely agreed):** "these demos are a bit disingenuous… they're not actually passing the game frames to Jev, they're passing a significantly smaller game state." Adam: "totally, worth noting." TypeSafe's own nuance says the same: "structured state as a data structure with text, not on images (yet…)".

### No-LLM chatbot (video)

- CJ's smart-home assistant: Jev decides *request/confirm/cancel*, *which tool* (units, web search, Wikipedia, recipes, todo, Home Assistant), and **extracts arguments from the state** ("which city" → "Denver"; "point to the line in the Wikipedia article that answers the question"). 300 ms from utterance to Home Assistant call. Every answer is grounded in a tool result; no generated prose.

### Verification / guardrails

- Syntax podcast: LLM writes show notes, then Jev checks each extracted claim against the transcript (Noul per claim).
- Code review risk matrix per modified file; jailbreak/prompt-injection detection (Gabriella asked whether injection data was in the training set — unanswered).

### The recurring summary

> "jev is just a **really** smart switch statement" — and the advice that follows: "don't expect magic on complex reasoning, it shines on classification/routing tasks." (Adam, Vibey)

---

## 3. Could GLiNER2 / 2.5 do this job?

### Mapping the Jev API onto GLiNER

| Jev primitive | GLiNER2 / 2.5 equivalent | Notes |
| --- | --- | --- |
| `state` (string / JSON / array) | the text after `[SEP]` | JSON must be serialised to text; DeBERTa was not pretrained on program state the way Jev was. Keep it short and flat (key: value lines). |
| **Noul** (yes/no with `p`) | single-label classification `{task: [yes, no]}` → `p(yes)`; or a 1-label multi-label task with sigmoid | `ClassAct::Softmax` / `Sigmoid` in `src/classification.rs` |
| **Choice** (≤255 options, probabilities + confidence) | single-label classification with N `[L]` labels, softmax over the `[L]` logits; compute `(p_max − 1/K)/(1 − 1/K)` yourself | each label + description consumes prompt tokens, so practical cardinality is tens, not 255; 2.5's constrained `Classifier` adds cross-task rules Jev doesn't have |
| **Score** (ordered levels, continuous position) | single-label classification over the level descriptions; `score = Σ i·p_i` | "describe situations, not degrees" advice transfers 1:1 — GLiNER supports label descriptions |
| Multiple questions per state, parallel | multiple classification schemas in **one** forward pass | already how `extract_internal` in `pipeline.rs` works (`schema_tokens_list`) — same cost profile as Jev: one encode, many cheap readouts |
| Argument / quote extraction ("which city", "point to the line") | **entity extraction** (2.0: ≤8-word spans; 2.5: any length) | a real advantage: GLiNER extracts spans natively; Jev's extraction is a pointer over pre-segmented state |
| Structured record from state | `extract_json` / records | no Jev equivalent |
| Relations / graph | relations, 2.5 `JointIE` | no Jev equivalent |
| Calibrated probabilities | **not trained for calibration** | sigmoid/softmax of a supervised MLP; needs post-hoc calibration (temperature/Platt on a held-out set) or conformal thresholds |
| Type safety | same guarantee — outputs are only the labels/spans you declared | |

### Zero-shot GLiNER2 / 2.5 in a Jev-shaped role: what to expect

**Works well out of the box**

- Lexically grounded routing and classification: team/category/intent labels with descriptions, spam/priority, topic. Published zero-shot F1: CLINC intent 62–64, AG News ~70, IMDB 86–90, Few-NERD 47–55.
- Multi-question per call at ~10–60 ms on CPU for short state (DeBERTa-base, ONNX in this crate) — latency comparable to or better than Jev's 70–500 ms, with no network hop, no waitlist, no data leaving the box (answers Nafiz's confidential-data and Wieiwowk's training-on-user-data concerns outright, and removes Lukas's distillation-attack worry).
- Argument extraction and grounded pointers (city, dates, quoted sentence) — better than Jev, since extraction is GLiNER's core skill.
- Compaction-style scoring (tamara's `p(drop)`): batch each transcript line as state with a Noul-like `[drop, keep]` task. Cheap enough to run on every tool call.

**Where it will be noticeably weaker than Jev zero-shot**

- Judgement questions that need world knowledge or multi-step inference ("does this need attention *right now*", "is this tool call risky", "how strong is the causal evidence", "long or short"). A 200M encoder trained on entity/classification data does not have Jev's "cognitive core"; expect flatter, less reliable probabilities.
- Calibration. Raw GLiNER probabilities are not calibrated; thresholds like "auto-act above 0.9, review below 0.5" must be set from your own eval set. Vaibhav's "slap a BERT in front" is right that it works, but the confidence is the part that needs engineering.
- Long / structured state. GLiNER2 base is trained to a few hundred words; 2.5 to 4096 words. Jev accepts 32k tokens per branch. Filtering state in code first (which Jev's docs also insist on) matters even more.
- High-cardinality Choice (hundreds of options, Wikiracing-style). GLiNER labels live in the prompt; 255 labels with descriptions will not fit well. 2.5's constrained `Classifier` helps with structure but not scale.
- Ordered `Score` with a continuous position — approximated, not native.

### With one or more fine-tuned LoRAs

This is where GLiNER has a structural advantage: **TypeSafe offers no fine-tuning** (asked at least three times in the thread, no answer), whereas `gliner2` ships `training/` with LoRA (`gliner2/training/lora.py`), and this crate already supports **hot-swapping merged adapter encoders** (`tutorial_11_adapter_switching`, `src/adapters.rs`, `export_adapter_encoder.py`). Note the crate's `training.rs` is a `todo!()` stub — training happens in Python; the crate consumes exported ONNX.

Recipe that maps directly onto what people reported working:

1. **Distil from a frontier model** (Arc: "train smaller models on Bedrock distillation, 100% synthetic from the teacher, barely any accuracy loss"; Aaron already has an eval set of Astra/Fable triage decisions). Run your existing LLM prompt — or Jev itself, via the Gateway — over a few thousand real states; keep the labels *and* the probabilities.
2. **Train one LoRA per task family** (triage, model routing, tool-risk, compaction) on the classification/extraction schema you will use at inference. GLiGuard and the PII models are exactly this pattern on GLiNER2. Keep the base encoder frozen so adapters stay swappable (this crate's O(1) `load_adapter` / `unload_adapter`).
3. **Calibrate** on a held-out slice: temperature scaling per task, or conformal prediction for guaranteed-coverage "escalate" sets (Benoit's TimesFM/conformal comparison is the right instinct). Report ECE alongside accuracy — TypeSafe's calibration is the thing users actually pay for.
4. **Wire the System 1 → System 2 loop** in code, exactly as Foreman does: GLiNER decides; below-threshold or flagged → reasoning model; still uncertain → human. Log every decision + probability so the next distillation round has data (Lakshmi's "use the classifier to score synthetic data" closes the loop).

Expected outcome: on a *specific* workflow with a few thousand distilled examples, a LoRA'd GLiNER2 base should match or beat the zero-shot Jev numbers Aaron reported ("as well if not better than our Astra/Fable prompts"), because it is trained on the exact distribution. It will not match Jev's breadth — Jev is a general decision model; a LoRA'd GLiNER is a set of narrow ones. That is fine for the triage/routing/compaction/guardrail uses above, which are inherently per-product.

### GLiNER2 vs 2.5 for this use

- **Classification-only workloads (routing, triage, compaction, guardrails):** GLiNER2 is enough today in this crate, and its intent/sentiment numbers are marginally better. Start here.
- **Add 2.5 when** the state is long (transcripts, full tickets with history, tool outputs), when you want per-span attributes ("this sentence is the risky part"), when you need constrained multi-task decisions (harm-type only if unsafe; see `Classifier` + `constraints`), or when quotes/arguments can be long spans. XNLI +5.5 (base) / +24.75 (multi) suggests 2.5 is also better at entailment-style "does the state support this claim" checks — the Syntax claim-verification use case.

### Honest comparison

| | Jev | GLiNER2 / 2.5 (+ LoRA) |
| --- | --- | --- |
| Breadth / zero-shot judgement | frontier-ish, general | narrow; good on lexical tasks, weak on reasoning-ish judgement |
| Calibration | trained-in (RLCD), verified ECE ≈ 0.03 on MMLU | must be added post-hoc per task |
| Latency | 70–500 ms, network | 10–60 ms CPU, local |
| Cost | $0.042/M tokens | hardware you already have |
| Fine-tuning | none | LoRA, swappable per task |
| Privacy | ZDR flags; data leaves your network | fully local |
| Extraction | pointer over pre-segmented state | native spans, records, relations |
| High-cardinality choice | ≤255 | tens |
| Structured state | trained on program state | serialise to text; untested |
| Multimodal | no | no |
| Availability | waitlist or Vercel Gateway | Apache-2.0 weights, this crate |

---

## 4. Suggested experiment

1. Take Aaron's shape: feedback card → `{disposition: [accept, reject]}`, `{priority: [low, medium, high]}`, `{needs_reasoning: [yes, no]}` as three classification schemas in one `extract` call on GLiNER2 base via this crate. Measure latency and raw agreement against a Jev / frontier-model labelled set of a few hundred cards.
2. Fit temperature per task; plot reliability diagrams; pick escalate thresholds.
3. Distil ~2–5k examples from the same teacher; train a LoRA in Python; export merged `encoder.onnx`; re-run 1–2 with `load_adapter`.
4. Repeat one extraction-heavy case (tool arguments / quoted evidence) to show where GLiNER beats Jev outright.
5. If long state or attributes matter, that's the trigger for the 2.5 port in the companion doc.

---

## 5. The open "Jev-alikes" (openjev.com / SemIf) and Rust

[openjev.com](https://openjev.com/) (renamed **SemIf**, independent, not
affiliated with TypeSafe) ships **no model of its own**. It is a browser demo
(wllama = llama.cpp in WASM + WebGPU) over stock GGUF checkpoints —
`Qwen/Qwen3-0.6B` (Q8_0, 639 MB), `openbmb/MiniCPM5-2B` (Q4_K_M, 1.56 GB),
`bartowski/Qwen_Qwen3.5-4B` (Q4_K_M, 3.01 GB). The mechanism (`worker.js`):

1. Prompt: `State: … / Question: … / Allowed options: A. … B. … / Reply with
   exactly one option letter`.
2. One prefill, `max_tokens: 1`, grammar-constrained to the letters, `top_logprobs: 20`.
3. Pick the letter tokens' logprobs (token id of "A" = 32 in Qwen, 54 in
   MiniCPM), softmax over just those → a Choice distribution. No decoding.
4. A second lane generates the same distribution as JSON text to show it is
   5–10× slower.

Their numbers on a 102-row TypeSafe subset (agreement): Qwen3.5-4B 84.5 %,
MiniCPM5-2B 63.7 %, Qwen3-0.6B 40.7 %, published Jev 88.3 %. Probabilities are
explicitly *not* calibrated. Siblings: `openjev-score` Python CLI (MIT, CUDA,
Qwen3.5-4B BF16), mini-jev (Qwen3-4B, HTTP), jevlike (a small encoder you train
— the GLiNER idea), an MLX parallel-constrained-decoding engine.

### Rust support

Straightforward, and cleaner than the JS because we can read logits directly:

- **Backend:** `llama-cpp-2` (same GGUFs, Metal/CUDA/CPU) or `mistral.rs` /
  `candle`. Prefill → `get_logits_ith(last)` → index letter token ids (looked
  up via the tokenizer at runtime, so any instruct GGUF works) → softmax.
- **State once, many questions** (Jev's packing property) via the KV cache:
  prefill the state into one sequence, `kv_cache_seq_cp` per question, prefill
  only question + options. The browser demo cannot do this (`cache_prompt: false`).
- **Primitives:** Noul = `[yes, no]`; Score = levels as options + expectation
  over the distribution. Choice cardinality bounded by single-token labels
  (~20–50); beyond that shortlist-then-classify, as Jev itself does above 255.
- **Cheap quality wins the demo skips:** permute option order and average
  (letter-position bias); temperature scaling on a labelled set.

### Relation to this crate

A 0.6–4B decoder is a different animal from a 194M encoder, so it belongs
*beside* GLiNER, not inside it:

| | GLiNER2/2.5 (this crate) | logit readout (llama.cpp) |
| --- | --- | --- |
| Zero-shot judgement | weak on reasoning-ish questions | 4B within ~4 pts of Jev on their set |
| Latency, CPU | 10–60 ms | ~1–3 s / 500-token prompt at 4B Q4 (≈100 ms on GPU/Metal) |
| Context | 512 tok / 4,096 words | 32k+ native (matches Jev) |
| Spans / records / relations | native | no |
| Calibration | add temperature | add temperature |
| Fine-tuning | LoRA in Python, hot-swap ONNX | LoRA possible, heavier |

Proposed shape: a small `System1` trait (`choice`, `noul`, `score` →
distribution + confidence) with `GlinerBackend` (this crate) and
`LogitReadoutBackend` (separate crate, e.g. `semif-rs`, exposed via an optional
feature so GLiNER users do not pull in llama.cpp). Use the readout backend as a
"System 1.5" tier between GLiNER and a frontier model in the escalation
ladder, or route per question. Effort: ~2–3 days for the readout crate with
KV-shared multi-question, permutation averaging and temperature scaling; ~1 day
for the trait and GLiNER impl.
