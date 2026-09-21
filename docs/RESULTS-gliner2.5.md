# GLiNER2.5 validation and benchmark results

**Status: pre-release M7 working document.** M0–M6 and the explicit-span API are
accepted on the pinned base checkpoint. Fresh small/base/multi bundles have
passed seven source/ONNX stages and native 30+2-case parity under the unchanged
numerical gates. Validated bundles are published at immutable Hugging Face revision
`27310cd26099a387b9936a1e13b03d6a0700baf2`; both actual downloaders passed clean
readback for all five profiles. Remote CI and the clean pushed-source downstream
consumer pass. Authoritative latency measurements and the release tag remain pending.

## Reproducibility scope

| Item | Value |
| --- | --- |
| GLiNER2 source | `d7c727458bf6929bc9ef5ee04e13c3f717a7c455` |
| Reference model | `fastino/gliner2.5-base-v1` at `78cea040597df251eedefa9d7ee2a756af39fe64` |
| Reference platform | macOS arm64, CPU fp32; pinned Torch reference uses one thread |
| Rust runtime | direct `ort = 2.0.0-rc.13`, native ONNX Runtime 1.28, four intra-op threads, Level3 optimization |
| Rust toolchains verified at M7 | 1.95 and minimum supported 1.91; 245 strict tests each, zero skips |
| Python reference environment | `scripts/export/env`, Torch 2.8.0, ONNX Runtime 1.20.1 |
| Export | ONNX opset 17, fp32 |
| Candidate-pool frozen reference | pinned PyTorch 2.8 CPU fp32 AArch64 arithmetic/order |

These results establish a bounded CPU/reference contract. They are not a claim
of bit identity on every CPU architecture, execution provider, ONNX Runtime
version or regenerated host-native Torch oracle. Small and multi have their own independent source, graph and native validation;
base measurements are not reused or relabeled for them.

The usual numerical gate is:

```text
abs(actual - reference) <= 1e-4 + 1e-3 * abs(reference)
```

The only boundary-stage adjustment is the reviewed centered `inside_prefix`
rule:

```text
abs(actual[i] - reference[i])
  <= 1e-4 + 1.1e-6 * i + 1e-3 * abs(reference[i])
```

where `i` is the boundary coordinate. Scorer and final-output tolerances were not
increased: final confidence absolute error remains at most `1e-3`, while pool
indices/order/masks and formatted non-confidence outputs are exact. The direct
ORT migration has a separate v2-only exception of at most `1e-6` absolute on
finite confidence values; labels, text, spans, ordering and other values remain
exact.

## Accepted base-checkpoint evidence

The following numbers are copied from the reviewed evidence files; they are not
latency benchmarks.

| Surface | Coverage | Maximum observed error / discrete result | Evidence |
| --- | --- | --- | --- |
| Encoder | 34 inputs, including all 30 corpus cases and dynamic batch/length cases | max absolute `6.4440072e-5`; combined original gate passed | [`evidence/m1.json`](evidence/m1.json) |
| Classifier | 20 raw/temperature-adjusted comparisons | max absolute `2.861023e-6` | [`evidence/m1.json`](evidence/m1.json) |
| Marginals | 24 real head cases plus bypass/dynamic/batch cases | non-prefix maxima up to `5.722046e-6`; prefix max `0.0031368732` under the coordinate rule | [`evidence/m2.json`](evidence/m2.json) |
| Marginal sensitivity | 5,116 valid candidate/query pairs | normalized inside and pair-logit max `6.103516e-5`; confidence max `8.6426735e-7`; no top-1 or 0.5 crossings | [`evidence/m2.json`](evidence/m2.json) |
| Rust candidate pool | 24 frozen real + 9 synthetic cases | frozen indices/order/mask and proposal arithmetic exact; actual ONNX-to-pool discrete outputs exact on 24 cases | [`evidence/m3.json`](evidence/m3.json) |
| Shared scorer | 24 real + dynamic/batch cases | pair logit max `9.346008e-5`; candidate state `9.536743e-7`; null `1.430511e-6`; count `4.172325e-7` | [`evidence/m4.json`](evidence/m4.json) |
| Entities/classification | 21 corpus + 2 independently regenerated mixed-task cases | labels/text/order/UTF-8 offsets exact; max confidence error `5.7816505e-6` | [`evidence/m4.json`](evidence/m4.json) |
| Records/JSON | 5 structure cases | shape/text/order/UTF-8 coordinates exact; reported record max confidence error `3.6805868e-6` | [`evidence/m5.json`](evidence/m5.json) |
| Relation graph | 4 real + 5 synthetic/raw probes | graph max absolute `1.0490417e-5`; graph-derived confidence max `1.0430813e-6` | [`evidence/m6.json`](evidence/m6.json) |
| Final relations | 4 relation-only + 2 Unicode/mixed-task oracles | non-confidence output exact; max confidence error `8.9406967e-7` | [`evidence/m6.json`](evidence/m6.json) |
| Explicit spans | 3 independent English/Unicode cases | ordered bounds/text exact; original logit gate and `1e-3` confidence gate passed | [`evidence/public-explicit-spans.json`](evidence/public-explicit-spans.json) |
| v2 direct-ORT regression | 6 original tutorials | one confidence changed by `5.364418e-7`; all other normalized output exact | [`evidence/ort-migration.json`](evidence/ort-migration.json) |

M6's parent gate ran 217 strict model-backed tests with no artifact skips on both
Rust 1.95 and 1.91. The corresponding no-model run also reported 217 passing test
cases, **but emitted 47 model-artifact skip messages**. Therefore the no-model
count proves model-free behavior and skip hygiene; it does not mean all 217 tests
executed model inference. The six v2 tutorial comparator found one confidence
difference (`5.364418e-7`) and exact normalized non-confidence output.

Earlier M0–M4 evidence was produced before the accepted direct-runtime migration
and records its historical ORT rc.9/native 1.20 context. It is not relabeled as
rc.13 evidence. M5, M6, explicit-span and migration gates separately exercise
the direct rc.13/native 1.28 runtime.

## Fresh three-checkpoint bundle validation

Each checkpoint used independently generated upstream outputs:30 original corpus
cases, one Unicode mixed-task case and one explicit duplicate-span case. All
non-confidence values and UTF-8 coordinates match exactly through native ORT1.28.

| Checkpoint | Source/ONNX stages | Native cases | Maximum confidence error | Maximum explicit-logit error |
| --- | ---: | ---: | ---: | ---: |
| small | 7/7 | 32/32 | `3.159046e-6` | `1.311302e-5` |
| base | 7/7 | 32/32 | `3.680587e-6` | `7.152557e-6` |
| multi | 7/7 | 32/32 | `1.430511e-5` | `3.671646e-5` |

The first multi run failed the centered-prefix gate. The correction preserves
fp32 and the pinned AArch64 PyTorch reduction order rather than relaxing the
bound: four interleaved four-lane accumulators with a four-level cascade,
exported using dynamic ONNX control flow. Masking, count/division, centering and
cumsum remain unchanged. Tiny-graph tests reproduce210 saved sums/means exactly
and cover68 synthetic cases through65,537 tokens. Full marginal gates then pass
for all three checkpoints; their other six graph hashes remain unchanged.
See [`evidence/m7-integration.json`](evidence/m7-integration.json).

Unsupported native empty-axis diagnostics are now opt-in using
`--probe-unsupported-axes`, because known Python ORT1.20.1 SIGSEGVs trigger macOS
crash dialogs. Default reports explicitly mark those probes not run. Supported
inputs and native Rust caller-rejection/bypass checks remain mandatory.

## Published artifacts and final native gates

All three manifests were promoted only after review of the actual seven-stage
and native reports, unchanged tolerances, source pins, recursive graph audits,
and streamed file hashes. The public Rust `validate_bundle` accepted all three:
16 authenticated files plus the manifest in each directory. Each includes a
sanitized validation report and the original unvalidated export manifest.

Publication: [`codesoda/gliner2-onnx@27310cd`](https://huggingface.co/codesoda/gliner2-onnx/commit/27310cd26099a387b9936a1e13b03d6a0700baf2).
The commit adds 52 files (three 17-file bundles and the root model card); no legacy
v2 graph was changed. Manifest SHA-256 values:

| Profile | Manifest SHA-256 |
| --- | --- |
| small | `5c691ff523278268f9046a42e8107c91fc084514f7f3f91cf88fc2fb9eb3e905` |
| base | `b986177cfc3a14e388d03d79b18b040e5873c47ad713e2b68396d3abfbc4ec5c` |
| multi | `ff51c54d0df6b870d204ee569dd6379d6607b43652dc142d1b4433dd277f3179` |

Actual Python and Rust `--model all --revision 27310cd26099a387b9936a1e13b03d6a0700baf2`
downloads used separate initially absent destinations and isolated caches. Both
returned zero and produced exactly 64 regular, nonsymlink files totaling
5,150,060,191 bytes each. Every path, size and SHA-256 matched, including original
v2 graph/metadata pins. Parent review independently rehashed all 128 destination
files. Readback report SHA-256:
`b37b8b6f682085f0efa2a1e20943ab4124b962c616353665ce41d27d80f26a15`.

Final corrected-artifact gates passed on both Rust 1.95 and 1.91: **245 tests,
zero failures and zero artifact skips** each; fmt and all-target/all-feature
Clippy with warnings denied also passed. The separate no-model gate reports
245 passing cases with **48 expected artifact skips**, not 245 inference tests.
All six fresh v2 tutorials passed with the same sole confidence drift
`5.364418029785156e-7`; other normalized output remains exact.

[Remote CI run 35563422282](https://github.com/codesoda/gliner2-rs/actions/runs/35563422282)
passed all three jobs at source `9332af6f10607a161ebfb8eb2a75051732e0495c`.

A fresh external consumer pinned that exact remote Git SHA, generated its own
Cargo.lock in an empty Cargo home, built on Rust 1.91, and ran against the fresh
Rust-downloaded base v2 and 2.5 bundles. All 24 files were independently bound to
immutable remote identities before the build. Selected task-family, records,
options, batch, explicit-span and v2 compatibility smoke checks passed.

Cargo/build/inference descendants ran under macOS denial of Python-prefixed
executable basenames, including absolute/framework paths; the absolute-path
preflight failed with the expected permission denial. This is evidence of a
Python-independent build and inference path, not proof that the machine lacks
Python or a defense against renamed/embedded interpreters. The smoke is
structural/runtime coverage, not an accuracy assertion or exhaustive API test.
See [`evidence/m7-consumer.json`](evidence/m7-consumer.json) for checked source,
artifact identities and raw evidence hashes.

## CPU latency benchmark protocol

Authoritative timing is deferred until competing CPU-heavy activity is idle.
The release benchmark binary has been built, but a persistent headless Chrome
process group consuming roughly eight cores currently prevents a quiet run. The final run must use complete, independently validated base v2 and base
2.5 bundles and record the following before values replace `PENDING`:

- exact Git commit and clean/dirty state;
- model repository/revision and SHA-256 of every graph used;
- CPU model, core counts, RAM, OS/architecture and power mode;
- `rustc`, crate profile, `ort` crate and native ONNX Runtime versions;
- execution provider, intra-op/inter-op thread settings and graph optimization;
- benchmark command, schema/labels, threshold and deterministic text generator;
- actual word and tokenizer/subword counts for each input;
- warm-up count, measured repetition count and whether model construction is
  excluded from timed inference;
- per-run raw samples or a machine-readable report, not only an aggregate.

### Comparison workload

The comparison uses base v2 and base 2.5 on the same CPU with one fixed entity
schema and deterministic inputs of 50, 500 and 3000 words. Model loading is
measured separately from steady-state inference. Runs are sequential—no competing
export, validation or benchmark processes. The planned minimum is 3 untimed
warm-ups and 10 measured inference repetitions per model/length; if resource
constraints require a different count, the actual count must be recorded rather
than silently changed. Report median and distribution tails from raw durations.
Peak memory should be measured by one documented method and reported as process
RSS (or clearly labeled if a different metric is used).

The harness is implemented in `scripts/benchmark_gliner2.sh` and
`examples/benchmark_gliner2.rs`; its 11 model-free tests pass. The wrapper uses a
fresh process per model and records `/usr/bin/time` whole-process peak RSS,
raw timing samples and actual tokenizer counts. See
[`BENCHMARK-gliner2.5.md`](BENCHMARK-gliner2.5.md) for its invocation. The actual
measured report remains pending; compilation or `--help` is not timing evidence.

### Pending latency table

| Architecture/model | Words | Subwords | Warm-ups | Measured runs | Load time (ms) | Inference median (ms) | p90 (ms) | p95 (ms) | Min–max (ms) | Throughput (words/s) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| GLiNER2 base | 50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GLiNER2.5 base | 50 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GLiNER2 base | 500 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GLiNER2.5 base | 500 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GLiNER2 base | 3000 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| GLiNER2.5 base | 3000 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

No speedup, memory reduction or long-input completion claim should be made until
these fields and the raw report are reviewed. The boundary candidate cap does not
remove dense encoder/boundary operations, so end-to-end complexity must be
measured rather than inferred.

## M7 results still required

| Gate | Current result |
| --- | --- |
| Complete seven-graph small bundle, source proof and ONNX/native validation | Passed seven-stage and native 30+2-case checks; published |
| Complete seven-graph base bundle rebuilt through final bundle tooling | Passed seven-stage and native 30+2-case checks; published |
| Complete seven-graph multi bundle, own tokenizer/source proof and validation | Passed seven-stage and native 30+2-case checks after source-ordered prefix correction; published |
| Manifest promotion to `validated` + `release_ready: true` | Passed parent adjudication and actual Rust manifest verification, all three profiles |
| Hosted Hugging Face immutable revision and clean download/readback | Passed at `27310cd26099a387b9936a1e13b03d6a0700baf2` |
| Rust and Python selector/hash/path validation against published files | Passed both actual `all` downloads; identical 64 files / 5,150,060,191 bytes |
| Fresh v2 metadata colocation and legacy split-layout fallback | Both fresh colocated downloads verified; legacy six-tutorial regression passed |
| CPU benchmark table above | **PENDING quiet machine** |
| Remote CI at pushed commit | Passed run `35563422282` at `9332af6`; final release revision must also pass |
| External Rust consumer from pushed source, Python execution denied | Passed clean Cargo/build/inference proof at `9332af6`, Rust 1.91; bounded scope documented above |
| Matching version and GitHub release tag | **PENDING; no tag claimed** |

Optional M8 helpers—attributes, constrained classification, JointIE and
long-document chunk/merge APIs—are outside this result set and unsupported unless
they receive separate implementation and evidence.
