# Agentic / tool-calling classification use cases — catalogue

_Consolidated from the Jev launch material, the Syntax "wtf is jev?" video, Diogo
Almeida's X thread, the WhatsApp group screenshots and community posts (Sept
2026). Companion to [`jev-and-gliner.md`](./jev-and-gliner.md) and
[`gliner2-vs-gliner2.5.md`](./gliner2-vs-gliner2.5.md)._

Every scenario below is "one state, one or more cheap typed questions". For each
we note where it came from, how Jev frames it, how GLiNER would frame it, and
whether a fine-tuned GLiNER can realistically do it.

**Legend**

- **GLiNER fit:** ★★★ natural fit (often *better* than Jev, because the answer is
  a span in the input) · ★★ classification-shaped, works with a LoRA · ★ needs
  reasoning / world knowledge, expect a filter not a judge
- **Version:** 2.0 = works in this crate today · 2.5 = needs long context,
  explicit-span scoring, attributes or constrained decoding (see port plan)
- **FT (fine-tune) outlook:** what a per-task LoRA + calibration realistically
  gets you

---

## 1. Triage / escalation ladders (System 1 → System 2 → human)

The canonical pattern. A fast classifier handles the bulk; code checks
probabilities against thresholds; uncertain or flagged items escalate to a
reasoning model; whatever that can't settle goes to a person.

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Feedback / issue card triage: accept/reject, priority, evidence, lifecycle, counterargument | Aaron Vanston, BuildPass "Foreman" (WhatsApp + Loom) | 1–4 calls: Choice + Score + Nouls | 3–5 classification schemas in one `extract` | ★★ | 2.0 | High — Aaron already has an Astra/Fable-labelled eval set to distil from |
| 1.2 | Support ticket: team, urgency, frustration | TypeSafe docs hero example; video | Choice + Noul + Score | `{team:[…]}`, `{urgent:[yes,no]}`, `{frustration:[calm, annoyed, angry]}` | ★★ | 2.0 | High |
| 1.3 | Inbox: priority, spam, should-I-reply | Vogle demo (video) | 3 questions | 3 schemas | ★★ | 2.0 | High |
| 1.4 | Approve / deny a request in an automated workflow | InfoWorld summary | Noul | binary classification | ★★ | 2.0 | Medium — depends how policy-heavy |

**Deep dive — "Jev 80% → Luna → human" (Foreman).**
Tier 1: every fresh Linear card gets a snapshot; code gathers context; Jev
classifies via AI Gateway; code evaluates escalation conditions (probability
thresholds, contradictory answers, always-review categories). ~80% of cards are
confident enough and are applied after schema/decision checks and a Foreman CLI
dry-run. Tier 2: GPT-5.6 Luna (run locally through a Codex process on a ChatGPT
subscription) authors or reviews the judgment only for flagged cards. Tier 3:
human. A sentinel polls every five minutes; empty queue = zero model calls.
Reported as "as well if not better" than their all-frontier prompts on their eval
set, at a fraction of cost/latency. **The calibrated probability is what makes
the 80/20 split safe** — with GLiNER, the thresholds must be fitted on a labelled
set (temperature scaling / conformal) before the ladder is trustworthy.

## 2. Model / intent routing inside an agent

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2.1 | Per-turn model choice: "how much reasoning is required?" + "which profile?" | Sentry "Junior" Slackbot (video); LangChain `ModelRouterMiddleware` | Score + Choice | `{reasoning:[none, light, heavy]}`, `{profile:[…]}` | ★★ | 2.0 | High for profile; medium for reasoning-depth (fuzzy but learnable from routed turns) |
| 2.2 | Intent routing in front of a support flow: DB lookup vs LLM vs human | Flavio Copes ("the one most early experiments land on"); video | Choice | intent classification | ★★ | 2.0 | High |
| 2.3 | Routing every request where a small LLM router was too slow | Ben Field (WhatsApp) | Choice | classification, local CPU | ★★ | 2.0 | High |

## 3. Tool selection + argument filling (no-LLM agent)

Where GLiNER has a genuine edge: tool choice is classification, arguments are
**spans in the utterance**.

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | Request / confirmation / cancel | CJ's smart-home demo (video) | Choice | `{turn_type:[request, confirm, cancel]}` | ★★ | 2.0 | High — train on your own utterances for ambiguous phrasing |
| 3.2 | Which registered tool fulfils the last message (units, web search, Wikipedia, recipes, todo, Home Assistant) | video | Choice over tool list | `{tool:[…]}` with descriptions | ★★ | 2.0 | High; cardinality is tens, fine |
| 3.3 | Argument extraction: city, time (now/today/tomorrow), units | video (weather) | Choice + extract-from-state | entities `location`, `time`; `{units:[C,F]}` | ★★★ | 2.0 | Very high — native span extraction |
| 3.4 | Home Assistant control: device, action, brightness, colour ("turn off the top bulb", "increase brightness to 100%", "change to green") | video | Choice + extraction | `{action:[on, off, set]}` + entities `device`, `brightness`, `color` in one pass | ★★★ | 2.0 | Very high |
| 3.5 | Follow-ups using conversation history ("what's the high in Celsius" → units tool with 86°F) | video | history in state | history in state; extract `value`, `from_unit`, `to_unit` | ★★★ | 2.0 (short) / 2.5 (long history) | High |
| 3.6 | Grounded answering: point to the sentence in a retrieved page that answers the question | video (Mount Rainier / Wikipedia) | pointer over pre-segmented state | entity `answer_sentence` — 2.5 for arbitrary-length spans | ★★★ | 2.5 | High |

**Deep dive — CJ's no-LLM assistant.** Existing MCP tools, unmodified. One
call answers turn type, tool, and tool-specific arguments; code executes the tool
and returns its result — no generated prose. 300 ms utterance → Home Assistant
API. GLiNER equivalent returns tool choice *and* extracted arguments from a
single forward pass, locally, in tens of ms, with nothing leaving the house. Its
weak spot is fuzzy turn-type judgement, which a few hundred household utterances
in a LoRA fixes.

## 4. Guardrails and tool-call safety

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | Score a proposed tool call for risk; block before execution | LangChain `AutoModeMiddleware` | Score / Noul | `{risk:[safe, review, block]}` on serialised call | ★ / ★★ | 2.0 | Medium — lexical risk (rm -rf, payments) learnable; consequence reasoning not |
| 4.2 | Jailbreak / prompt-injection / malicious intent on inbound messages | video; Gabriella's question on X | Noul | classification (GLiGuard is exactly this on GLiNER2) | ★★ | 2.0 | High |
| 4.3 | Is the LLM output safe / on-topic / polite / correct before showing it | Outcome School; TypeSafe "verify everything" | Nouls | multi-task classification; 2.5 constrained `Classifier` for "harm-type only if unsafe" | ★★ | 2.0 / 2.5 | High for safety/tone; low for "correct" |
| 4.4 | PII detection & redaction across a whole document with offsets | GLiNER 2.5 release; Jev "verify everything" | n/a (Jev can't extract) | entities with char offsets, long-context | ★★★ | 2.5 | Very high (PII fine-tunes exist) |

## 5. Verification of LLM output

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | Claim-by-claim fact check of generated show notes against the transcript | Syntax podcast (video) | Noul per claim | `{supported:[yes,no]}` on (claim, transcript window); 2.5 for long transcripts and NLI gains | ★ / ★★ | 2.5 | Medium — NLI-shaped; 2.5 multi +25 on XNLI shows it's trainable |
| 5.2 | RAG passage relevance / citation check before anything expensive sees it | TypeSafe cookbooks; Valyu guide | Noul per passage | cross-encoder style `{relevant:[yes,no]}` per (query, passage) | ★★ | 2.0 | High — this is what DeBERTa rerankers do |
| 5.3 | Code review risk matrix per modified file (security, bad practice, complexity, commit message) | video | Scores | multi-task classification per file/diff hunk | ★ / ★★ | 2.5 | Medium — lexical parts yes, semantic parts no |
| 5.4 | Extract topics / tags / chapter markers and validate them | video | extraction + Nouls | entities + classification | ★★★ | 2.0 / 2.5 | High |

## 6. Context management

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6.1 | Instant compaction: `p(drop)` per tool call / transcript line | tamara (@tamarajtran), 1.7M views | one Noul per line over shared state | 2.0: per-line `(goal, line)` pairs → `{keep, drop}`; 2.5: one pass with `score_explicit_spans` over line boundaries | ★★ | 2.0 (prototype) / 2.5 (single pass) | Very high — labels are free from your own traces |
| 6.2 | Retrieve wide, filter cheaply, send shortlist to the LLM | literature-review example (Valyu) | Noul per item | classification per item | ★★ | 2.0 | High |
| 6.3 | Which prior turns / tool outputs to re-inject for the next step | implied by 6.1 | Nouls | same as 6.1 | ★★ | 2.5 | High |

**Deep dive — compaction.** tamara's screenshot keeps the *narrative* (tool call
headers, failures, decisions, TODOs, user instructions) and drops *payloads*
(file contents, verbose test output, intermediate greps) the model already acted
on and can re-fetch. Signals are structural, not semantic — exactly what a
cross-encoder learns easily. Training labels are nearly free: a line is `keep`
if any later turn referenced it (path, error string, identifier) or if the agent
re-read it after a naive compaction; optionally distil from a frontier model or
Jev; counterfactual-check a small eval set. Be conservative: dropping something
needed later costs far more than keeping junk, so calibrate thresholds and
replace dropped blocks with one-line stubs (`[dropped: 212-line read of
parser.ts]`) rather than deleting. "Relevant going forward" is a prediction
about the goal; both Jev and GLiNER only see the current goal. Zero-shot GLiNER
will be mediocre here — the LoRA is what makes it work.

## 7. Bulk classification / map-reduce over data

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7.1 | 1,000+ research papers by category (8¢ total, ~256 ms each) | video | Choice | classification; free locally | ★★ | 2.0 | Very high |
| 7.2 | Resume ↔ job-posting fit | Hiring Cafe / JMED (video) | Score | `{fit:[poor, partial, strong]}` on (posting, resume) — long inputs | ★★ | 2.5 | High |
| 7.3 | Integration category mapping | Jack McNicol (WhatsApp) | Choice | classification with label descriptions | ★★ | 2.0 | Very high |
| 7.4 | Unstructured-data classification replacing unreliable cheap LLMs | Stuart (WhatsApp) | Choice / Noul | classification | ★★ | 2.0 | Very high |
| 7.5 | Scoring synthetic data so LLMs generate better training sets | Lakshmi Narayanan (X) | Score | quality classification / rubric levels | ★★ | 2.0 | High |
| 7.6 | "Turn petabytes into features" | TypeSafe | many questions | many schemas per row | ★★ | 2.0 | Very high — cost is the whole point |

## 8. Real-time / interactive scoring

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8.1 | Live tone / conviction / urgency / "reads AI-written" as you type | video | Scores | multi-task classification per paragraph; local latency beats network | ★★ | 2.0 | High |
| 8.2 | Feed filter extension: hide rage-bait / crypto / politics | video | Nouls | classification per post | ★★ | 2.0 | Very high |
| 8.3 | Live debate "BS meter" per sentence | video | Noul | claim classification; fact-checking itself needs retrieval | ★ | 2.0 | Low for truth, high for "checkable claim?" |
| 8.4 | Game / state-driven control: Tetris, Doom, Subway Surfers, driving sim, Wikiracing | video; TypeSafe demos; Max Blade | Choice over actions on serialised state | classification over serialised state | ★ / ★★ | 2.0 | Medium — same caveat as Jev: text state, not pixels (ben's critique); Wikiracing needs shortlist-then-classify due to cardinality |

## 9. Structured decisions on program state

| # | Scenario | Source | Jev framing | GLiNER framing | Fit | Version | FT outlook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9.1 | Long / short from market features | Matt's question on X | Choice with probabilities | classification over serialised JSON | ★ | 2.0 | Low–medium — DeBERTa not pretrained on program state; per-schema LoRA only |
| 9.2 | Clinical: symptoms + medications with negation / dosage attached to each span | GLiNER 2.5 release | n/a | entities + span attributes | ★★★ | 2.5 | Very high |
| 9.3 | Contract / document: parties, obligations, clauses across 300 pages | GLiNER 2.5 release | n/a | long-context entities, records | ★★★ | 2.5 | High |
| 9.4 | Knowledge-graph construction for agent memory (people, projects, commitments) | GLiNER 2.5 release | n/a | entities + relations + `JointIE` | ★★★ | 2.5 | High |

---

## Summary

- ~35 scenarios across 9 shapes. Roughly 7–8 of the 9 shapes are viable for a
  fine-tuned GLiNER; the exceptions are high-cardinality choice (>~50 options)
  and anything that needs reasoning or world knowledge rather than a learnable
  mapping from state text to labels/spans.
- **GLiNER beats Jev outright** wherever the answer is a span in the input:
  3.3–3.6, 4.4, 5.4, 9.2–9.4. Jev cannot extract, record, or relate.
- **GLiNER matches Jev with a LoRA** on narrow, high-frequency decisions:
  triage (1), routing (2), tool choice (3.1–3.2), compaction (6), bulk (7),
  real-time (8.1–8.2), guardrails (4.2–4.3).
- **GLiNER stays behind Jev** on breadth and zero-shot judgement (4.1, 5.1, 5.3,
  8.3, 9.1) and on calibration unless you add it yourself.
- **Priority for the first experiment:** 1.1/1.2 (triage), 2.1 (routing), 3.4
  (tool + args), 6.1 (compaction). All are one-state-many-questions, all have
  cheap or existing labels, and 3.4/6.1 show the two things Jev can't do or
  can't do locally.
- **Trigger for the 2.5 port:** any of 3.6, 4.4, 5.1, 6.1 single-pass, 7.2,
  9.2–9.4 — long state, explicit-span scoring, attributes, or relations.
