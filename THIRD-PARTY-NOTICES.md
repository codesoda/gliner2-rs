# Third-party source notices

## GLiNER2 boundary inference

Boundary candidate selection, decoding, preprocessing, classification selection,
record assignment, and relation proposals/edge deduplication in `src/boundary/`,
together with the related Python
export/parity utilities, include modified adaptations of Fastino's GLiNER2
implementation at commit `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`.
The Rust ports replace tensor/control-flow operations while preserving the
pinned upstream inference behavior. Graph wrappers reuse upstream learned
modules; the validation oracle remains unmodified.

Upstream: <https://github.com/fastino-ai/GLiNER2>.
The upstream Apache License 2.0 is reproduced in
[`docs/licenses/GLiNER2-Apache-2.0.txt`](docs/licenses/GLiNER2-Apache-2.0.txt).

## LLVM libc++ sorting algorithms

The unstable argsort/introsort and heap-fallback routines in
`src/boundary/record_decode.rs` are modified Rust adaptations of LLVM libc++
algorithm implementations (`sort.h`, `partial_sort.h`, `make_heap.h`,
`sort_heap.h`, `pop_heap.h`, `sift_down.h`, and `push_heap.h`). They preserve
observable ordering from the pinned macOS arm64 PyTorch reference rather than
using Rust's platform-dependent unstable sort.

These portions originate in the LLVM Project and are distributed under the
Apache License 2.0 with LLVM Exceptions. The complete upstream libc++ license,
including its retained legacy notices, is provided in
[`docs/licenses/LLVM-libcxx.txt`](docs/licenses/LLVM-libcxx.txt).

Upstream: <https://github.com/llvm/llvm-project/tree/main/libcxx/include/__algorithm>.
License source: <https://github.com/llvm/llvm-project/blob/main/libcxx/LICENSE.TXT>.
This attribution does not assign a license to unrelated project code.

## SLEEF relation-proposal exponential

`src/boundary/relation_pairs.rs` contains a modified scalar Rust adaptation of
SLEEF's single-precision `expf` u10 operation solely to reproduce the pinned
macOS arm64 PyTorch 2.8 CPU vector-kernel rounding used to rank relation
proposals. This is a bounded reference-platform compatibility operation, not a
claim of universal SLEEF or arbitrary-platform bit parity.

The adaptation is based on SLEEF commit
`5a1d179df9cf652951b59010a2d2075372d67f68`, specifically
`src/libm/sleefsimdsp.c` (`xexpf`), `src/common/misc.h` (range-reduction
constants), and `src/arch/helperadvsimd.h` (fused multiply-add operation order).
Copyright Naoki Shibata and contributors 2010–2024.
The AArch64 helper also carries Copyright ARM Ltd. 2010–2024.
Upstream: <https://github.com/shibatch/sleef/tree/5a1d179df9cf652951b59010a2d2075372d67f68>.
The Boost Software License 1.0 is reproduced in
[`docs/licenses/SLEEF-Boost-1.0.txt`](docs/licenses/SLEEF-Boost-1.0.txt).

## Pinned choice-field Unicode behavior

`src/boundary/choice_unicode.rs` contains compact generated tables for Python
3.12.7 / Unicode 15.0.0 lowercase, casefold, regular-expression case equivalence,
and character properties. `scripts/parity/gen_choice_unicode.py` derives these
from the pinned CPython interpreter and validates them; Rust lookup routines
replace interpreter calls at inference time. The data representation is changed
and compressed, not a bundled Python runtime.

CPython's license and retained historical notices are reproduced in
[`docs/licenses/CPython-3.12.7.txt`](docs/licenses/CPython-3.12.7.txt), from
<https://github.com/python/cpython/blob/v3.12.7/LICENSE>.
The Unicode data copyright and permission notice is reproduced in
[`docs/licenses/Unicode-3.0.txt`](docs/licenses/Unicode-3.0.txt), from
<https://www.unicode.org/license.txt>.
