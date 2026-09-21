#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# No Cargo invocation, downloads, export, or model inference in Python. The
# standard-library wrapper launches one prebuilt Rust process per selected model,
# retains raw /usr/bin/time logs, and atomically publishes only complete evidence.
exec python3 - "$repo_root" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss(log, platform):
    if platform == "darwin":
        pattern = r"^\s*(\d+)\s+maximum resident set size\s*$"
        unit, multiplier = "bytes", 1
    elif platform.startswith("linux"):
        pattern = r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$"
        unit, multiplier = "KiB (GNU time labels kbytes)", 1024
    else:
        raise RuntimeError("RSS measurement supports macOS and Linux only")
    matches = re.findall(pattern, log, re.MULTILINE)
    if len(matches) != 1 or int(matches[0]) <= 0:
        raise RuntimeError("missing, ambiguous, or invalid maximum RSS in resource log")
    return {
        "raw_value": int(matches[0]), "raw_unit": unit,
        "bytes": int(matches[0]) * multiplier,
        "scope": "whole fresh Rust command high-water RSS under OS time accounting, including any accounted provenance helper children; across load, all sizes, verification and hashing; not per-case or model-only allocation",
        "method": "/usr/bin/time -l" if platform == "darwin" else "/usr/bin/time -v",
    }


def main():
    root = Path(sys.argv[1])
    parser = argparse.ArgumentParser(
        prog="scripts/benchmark_gliner2.sh", allow_abbrev=False,
        description="Run prebuilt CPU benchmark in one fresh Rust process per model. No build/download. "
                    "Requires Python 3 stdlib and /usr/bin/time. Set BENCHMARK_POWER_MODE to the observed "
                    "power/AC mode; BENCHMARK_BINARY can override the release executable path.",
        epilog="Other options are forwarded to the Rust example: --v2-model-dir, --v2-onnx-dir, "
               "--v25-bundle-dir, --warmups (default 3), --repetitions (default 10), --threshold, --label. "
               "Direct Rust --model both is non-isolated; this wrapper always isolates both.")
    parser.add_argument("--model", choices=("v2", "v25", "both"), default="both")
    parser.add_argument("--output", default=str(root / "benchmark-gliner2.json"),
                        help="new aggregate JSON path; existing paths are never overwritten")
    options, forwarded = parser.parse_known_args(sys.argv[2:])
    output = Path(options.output).absolute()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"report already exists: {output}; choose a new output")
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(root / "target")))
    binary = Path(os.environ.get("BENCHMARK_BINARY", str(target / "release/examples/benchmark_gliner2"))).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"prebuilt executable missing: {binary}; prebuild with cargo build --locked --release --example benchmark_gliner2")
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("resource collection supports macOS/Linux only")
    selected = ["v2", "v25"] if options.model == "both" else [options.model]
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix=output.name + ".evidence-", dir=output.parent))
    print(f"raw evidence directory (also retained on failure): {evidence}", file=sys.stderr)
    binary_sha = digest(binary)
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    reports = []
    resources = []
    for model in selected:
        raw_report = evidence / f"{model}.json"
        resource_log = evidence / f"{model}.resources.log"
        stdout_log = evidence / f"{model}.stdout.log"
        command = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v",
                   str(binary), *forwarded, "--model", model, "--output", str(raw_report)]
        print("running " + model + " in a fresh process; see " + str(resource_log), file=sys.stderr)
        with resource_log.open("xb") as stderr, stdout_log.open("xb") as stdout:
            subprocess.run(command, check=True, env=environment, stdout=stdout, stderr=stderr)
        report = json.loads(raw_report.read_text())
        expected_name = "gliner2-base-v1" if model == "v2" else "gliner2.5-base-v1"
        if (report.get("schema_version") != 2 or len(report.get("models", [])) != 1
                or report["models"][0]["name"] != expected_name):
            raise RuntimeError("unexpected per-process report identity/schema")
        if report["provenance"]["cargo_profile"] != "release":
            raise RuntimeError("authoritative wrapper requires a release build")
        report["models"][0]["peak_process_rss"] = peak_rss(resource_log.read_text(), sys.platform)
        if reports:
            baseline = reports[0]
            if report["texts"] != baseline["texts"]:
                raise RuntimeError("text identities differ across models")
            for key, value in baseline["configuration"].items():
                if key != "execution_order" and report["configuration"].get(key) != value:
                    raise RuntimeError(f"settings differ across models: {key}")
        reports.append(report)
        resources.append({
            "model": model, "command": command,
            "raw_report": str(raw_report), "raw_report_sha256": digest(raw_report),
            "resource_log": str(resource_log), "resource_log_sha256": digest(resource_log),
            "stdout_log": str(stdout_log), "stdout_log_sha256": digest(stdout_log),
        })
    if digest(binary) != binary_sha:
        raise RuntimeError("benchmark executable changed during run")
    aggregate = {
        "schema_version": 2, "benchmark": "cpu-end-to-end-entity-extraction-isolated",
        "invocation": [str(root / "scripts/benchmark_gliner2.sh"), *sys.argv[2:]],
        "binary": str(binary), "binary_sha256": binary_sha,
        "process_isolation": "one fresh Rust process per model, sequential; runtime first initialization isolated; filesystem caches NOT flushed; NOT disk-cold",
        "runs": reports, "raw_evidence": resources,
    }
    # Same-filesystem hard-link publication is atomic and refuses existing output.
    temporary = evidence / "aggregate.json.tmp"
    with temporary.open("x") as destination:
        json.dump(aggregate, destination, indent=2, allow_nan=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.link(temporary, output)
    temporary.unlink()
    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"benchmark failed; no aggregate report published: {error}")
PY
