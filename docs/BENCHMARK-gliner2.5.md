# GLiNER2 base vs GLiNER2.5 base CPU benchmark

`examples/benchmark_gliner2.rs` is an opt-in public-pipeline entity-extraction
benchmark. It never downloads models. This document specifies a procedure, not
results; no authoritative timing is claimed by implementing the harness.

## Fixed workload and tokenizer counts

Both models receive identical **50, 500, and 3,000 whitespace-delimited words**.
The generator cycles the following sequence and replaces the final word with `.`:

```text
Alice from Acme Corporation visited Paris on Monday . Bob joined Globex in Berlin on Tuesday . Carol met Dana at Initech in London
```

Each text's whitespace count, UTF-8 byte length and SHA-256 are recorded. Default
ordered labels are `person`, `organization`, `location`, `date`; threshold is
`0.5`. Repeated `--label` options replace the defaults. Word counts are fixed.
The wrapper checks matching text identities and inference settings across runs.

After all timings, each model's own `tokenizer.json` is loaded and hashed. The
report reproduces the public entity path's preprocessing and
`format_input_with_mapping` (per-word `RuntimeTokenizer.tokenize_token`,
`add_special_tokens=false`), recording:

- `retained_text_words`: words after the model's preprocessing/cap;
- `text_only_subwords`: formatted mappings whose segment is `Text`, excluding
  schema and separators;
- `schema_plus_text_subwords`: total formatted encoder input IDs, **including**
  schema and separator pieces;
- `max_original_words`: resolved config word cap (`null` means uncapped).

These are not raw whole-string tokenizer counts, nor whitespace counts renamed
as subwords. The tokenizer hash makes them reproducible. They are computed
outside measured calls rather than instrumenting inference.

### Truncation and window semantics

The span path uses the library regex word splitter, lowercases text tokens, and
applies `config.max_len` if present; the pinned legacy base config is verified
byte-for-byte. Boundary preprocessing preserves case, normalizes terminal
punctuation, uses the boundary whitespace splitter, and caps original split
words at `max_len` (default 4096). Schema/choice-prefix tokens are not included in
that original-word cap. Generated inputs already end with `.`. Both paths use
their existing pipeline behavior, not a benchmark-specific truncation override.

Neither path chunks these documents into sliding encoder windows or merges
chunk outputs. The boundary head's attention-window setting (128 in its graph
contract) is not a 128-word document cutoff or proof of linear memory usage.
Subword sequences can exceed the original-word cap. The full 3,000-word call
can be expensive: candidate caps do not remove dense encoder/boundary work.
Throughput is explicitly **submitted whitespace words / median call duration**,
not subwords/s, retained tokens/s, or proof that every submitted word survived
an arbitrary changed config. Consult retained counts before interpreting it.

## Models, immutable identities and layouts

The v2 split layout is supported:

- `--v2-model-dir`: config, tokenizer and tokenizer metadata;
- `--v2-onnx-dir`: `encoder.onnx`, `extractor_padded.onnx`, `classifier.onnx`.

For collocated v2 files, pass the same directory to both options. Metadata and
all three loaded graphs must match the compiled
[`checkpoints/v2-metadata-pins.json`](checkpoints/v2-metadata-pins.json) base
entries. The report binds `fastino/gliner2-base-v1` revision
`79c3a777abc572b4767922f3916cf63fb5754df2`, the hosted ONNX repository/revision,
the pin-document SHA-256, and actual metadata/graph sizes and hashes. The hosted
ONNX identity is distinct from the source-model revision; the harness does not
claim it independently re-exported those graphs.

`--v25-bundle-dir` requires a complete boundary bundle, `export_manifest.json`,
metadata/tokenizer and all seven loaded graphs:

```text
encoder.onnx
classifier.onnx
boundary_marginals.onnx
boundary_scorer.onnx
boundary_explicit_scorer.onnx
boundary_records.onnx
boundary_relations.onnx
```

The manifest must identify **base**, not small/multi:
`fastino/gliner2.5-base-v1` at
`78cea040597df251eedefa9d7ee2a756af39fe64`, upstream GLiNER2 commit
`d7c727458bf6929bc9ef5ee04e13c3f717a7c455`. Its exact bytes are snapshotted before
loading and SHA-256-bound in the report. After timings, every manifest-listed
file's size/hash is verified, required file entries and safe in-bundle paths are
checked, and config architecture/encoder identity is checked. A changed
manifest during the run fails closed. Do not modify any model files during a run.

An `exported-unvalidated` manifest with `release_ready: false` is accepted for
**development measurements only** and reported as such: checksums are not parity
validation or release readiness. A `validated`/`release_ready: true` manifest
also must pass the library's full `validate_bundle` after timings. Published
results require the latter; do not promote a manifest just to obtain timings.

Loading uses `SpanPipeline::new(...).with_classifier(...)` wrapped as
`AutoPipeline::Span` for v2 and `AutoPipeline::from_dir` for v2.5. Both timed
inference paths call `AutoPipeline::extract_entities`.

## Measurement and isolation

Defaults: **three warmups and ten measured repetitions per model/input size**.
`std::time::Instant` measures one successful public call, including preprocessing,
tokenization, encoder/head inference and decoding. Loading, output checks,
additional tokenizer counts, provenance hashing and serialization are excluded.
Every raw duration (`per_run_ns`) is retained, with median (integer average for
an even sample count), nearest-rank p90/p95, min, max, and submitted words/s at
the median. Small samples give coarse tail percentiles; p90/p95 may equal max.

Every nested result is consumed with `black_box`. Label-group order/count,
entity count, and a hash of labels/spans/text/confidence bits must remain stable
across warmup and measured runs per model/input. Cross-model output equality is
not required.

The authoritative shell wrapper starts **one fresh Rust process per model**,
sequentially, so neither the first ONNX Runtime initialization nor process peak
RSS is inherited from the other model. Rust `--model v2|v25|both` also permits
direct runs. Direct `both` runs v2 then v2.5 in one process and is explicitly
non-authoritative for initialization/memory comparisons.

Constructor timing is **warm-filesystem / non-disk-cold load time**, not disk-cold
latency. There is no cache eviction: previous validation, shared libraries, OS
caches and earlier processes can warm files. A fresh process isolates runtime
initialization, not disk caches. Graph hashing happens after all timings, but
that alone cannot establish cold storage. The boundary constructor loads seven
sessions; the v2 constructor loads three. Load times compare those constructors,
not just the entity head.

Sessions use four intra-op threads, CPUExecutionProvider and Level3 graph
optimization. Inter-op threads are **not explicitly configured**: the report
states ONNX Runtime default and records a null numeric value, not an invented
measured count. Physical cores are queried using `sysctl hw.physicalcpu` on macOS
or unique socket/core pairs from `lscpu -p=SOCKET,CORE` on Linux. Power/AC mode is
an **operator declaration** via `BENCHMARK_POWER_MODE`, not inferred telemetry.
Unavailable metadata is null.

### Peak process RSS

The wrapper uses `/usr/bin/time -l` on macOS or GNU `/usr/bin/time -v` on Linux.
It retains the raw stderr/resource log, extracts exactly one maximum-RSS value,
and records the raw value/unit and normalized bytes. macOS reports bytes; GNU
time's `kbytes` means KiB (multiply by 1024). Missing/ambiguous RSS fails the
aggregate report; it is never replaced with an estimate.

This is one **whole-process high-water mark per model**, spanning loading, all
three workloads, output checks, tokenizer recounting and provenance hashing.
It is not a per-length peak or isolated model-allocation count. OS `time`
accounting may include waited provenance helper children (such as `git` and
`rustc --version`); no subtraction is attempted. The Python orchestrator's
memory is outside the measured Rust command. Direct Rust reports
have `peak_process_rss: null`; the wrapper enriches each aggregate run with the
external measurement while preserving the original raw JSON.

## Quiet-machine authoritative run

Do not run alongside export, validation, another benchmark, or CPU/memory-heavy
work. Finish builds first, ensure stable power/performance settings and an idle
machine, then declare the settings actually observed:

```bash
cargo build --locked --release --example benchmark_gliner2
# Example declaration only: replace with the machine's actual observed setting.
export BENCHMARK_POWER_MODE='AC; Low Power Mode off; observed before run'
scripts/benchmark_gliner2.sh \
  --model both \
  --v2-model-dir models/gliner2-base-v1 \
  --v2-onnx-dir onnx/gliner2-base-v1 \
  --v25-bundle-dir onnx/gliner2.5-base-v1 \
  --warmups 3 \
  --repetitions 10 \
  --threshold 0.5 \
  --output /tmp/gliner2-base-vs-2.5-base.json
```

The wrapper requires Python 3 **standard library only** for subprocess/resource
log/JSON handling; Python does not perform model inference. The wrapper never
invokes Cargo or downloads anything. `CARGO_TARGET_DIR` selects the prebuilt
target directory, or `BENCHMARK_BINARY` names the release executable explicitly.
A non-release report is rejected. `--help` requires no models or executable.
Use a new output path per attempt. No automatic timeout is imposed. Smaller
explicit warmup/repetition counts are supported for development/resource-limited
runs and recorded honestly, but the planned publication minimum is 3/10.

## Machine-readable evidence and failure behavior

Both raw Rust reports and the aggregate use `schema_version: 2`. The aggregate
has `runs` (one complete Rust report per model), `raw_evidence`, the exact wrapper
invocation/child commands, binary SHA-256 and an explicit isolation statement.
Each run records:

- labels, threshold, word counts, warmups/repetitions, provider/thread settings,
  optimization, timing/load/resource scopes;
- text identities, tokenizer-scoped counts, load time, raw durations and summary
  statistics, deterministic output counts/hashes;
- source pins, manifest status/hash (v2.5), actual loaded graph sizes/hashes;
- git commit/dirty state, crate and locked `ort` versions, native ORT build info,
  Rust toolchain/profile, OS/architecture, CPU model, logical/physical cores,
  memory and declared power mode.

Raw JSON, stdout and resource logs live in a unique adjacent
`<output>.evidence-*` directory. Their paths and SHA-256 hashes are in the
aggregate. Preserve this directory with the aggregate and identify published
summaries by the aggregate's SHA-256.

Rust publishes JSON via a same-directory temporary and atomic no-clobber
persist. The wrapper publishes only after all selected runs and verification
succeed, using an atomic no-clobber hard link on the same filesystem. Existing
outputs are refused, never overwritten. On failure no new aggregate appears;
raw logs/partial-run evidence remain for diagnosis. Missing/wrong files,
identity mismatches, changing output, malformed CLI arguments and RSS parsing
failures are errors, not skipped models or placeholder timings.

## Model-free checks

Coordinate compilation with other agents; do not build or benchmark during
heavyweight validation. Static checks are safe:

```bash
rustfmt --edition 2024 --check examples/benchmark_gliner2.rs
bash -n scripts/benchmark_gliner2.sh
scripts/benchmark_gliner2.sh --help
```

When the build lane is idle, these tests require no model inference:

```bash
cargo test --locked --example benchmark_gliner2
cargo clippy --locked --example benchmark_gliner2 -- -D warnings
```

Tests cover deterministic inputs, statistics, CLI/model selection, identity and
path rejection, atomic report publication, SHA-256 vectors and locked ORT
version provenance. Neither a successful model-free test nor `--help` is timing
evidence.
