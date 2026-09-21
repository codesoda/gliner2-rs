#!/usr/bin/env bash
# Build and run a clean, out-of-tree Rust consumer against downloaded bundles.
# Model acquisition is deliberately separate and opt-in.
set -euo pipefail

DEFAULT_SOURCE="https://github.com/codesoda/gliner2-rs.git"
DEFAULT_HF_REPOSITORY="codesoda/gliner2-onnx"
DEFAULT_TOOLCHAIN="1.91.0"

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_external_consumer.sh \
    --rev <40-hex-pushed-git-sha> \
    --hf-revision <40-hex-downloaded-snapshot-sha> \
    --bundle-root <absolute-gliner2.5-bundle> \
    --v2-bundle-root <absolute-v2-bundle> \
    --report-dir <new-absolute-directory> [options]

Required:
  --rev SHA             Exact pushed source commit used by the git dependency.
  --hf-revision SHA     Immutable codesoda publication revision to verify by readback.
  --bundle-root DIR     Clean downloaded GLiNER2.5 bundle with export manifest.
  --v2-bundle-root DIR  Clean downloaded, colocated v2 bundle (no boundary manifest).
  --report-dir DIR      New directory outside this git checkout for immutable proof.

Options:
  --source URL          Git dependency URL (default: https://github.com/codesoda/gliner2-rs.git).
  --hf-repository ID    Must be codesoda/gliner2-onnx (the publication contract).
  --toolchain NAME      Installed rustup toolchain (default: 1.91.0).
  --work-dir DIR        New external work directory instead of a temporary directory.
  --keep-work           Retain the generated consumer, clean Cargo home, and target.
  --deny-python-exec    macOS: also deny execution of Python-named paths with sandbox-exec.
  -h, --help            Show this help.

Acquire fresh bundles separately with the explicit project downloader. This
script only reads remote HEAD identities and bounded small metadata (16 MiB/file,
64 MiB total); it never GETs ONNX graphs or acquires a missing bundle. All local
files must match immutable remote identities before Cargo or the consumer runs.
v2 metadata is verified against original fastino source pins, NOT the codesoda
publication revision; all v2 graphs in the pin contract are verified at codesoda.
The report retains checked identities, headers, manifests, pins, source remote
proof, Cargo.lock, and process output. Keep bundles unchanged during the run.
Default Python evidence is PATH-name blocking only, not an interpreter sandbox:
absolute paths, renamed interpreters and later PATH changes are not blocked.
--deny-python-exec additionally denies process-exec paths matching /python[^/]*$
(including absolute /usr/bin/python3) for Cargo and its descendants on macOS.
It does not deny renamed/embedded interpreters or model a malicious adversary.
Truly absent Python still needs a separate CI container. This smoke is not
exhaustive public-method coverage, an accuracy assessment, or a release-readiness
certification.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

canonical_dir() {
  local directory=$1
  [[ -d "$directory" ]] || die "not a directory: $directory"
  (cd "$directory" && pwd -P)
}

is_within() {
  local child=$1
  local parent=$2
  [[ "$child" == "$parent" || "$child" == "$parent"/* ]]
}

require_file() {
  local root=$1
  local relative=$2
  [[ -f "$root/$relative" ]] || die "bundle is missing $relative under $root"
}

sha256_file() {
  local file=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    die "sha256sum or shasum is required"
  fi
}

record_tree_checksums() {
  local root=$1
  local output=$2
  : >"$output"
  while IFS= read -r file; do
    local relative=${file#"$root"/}
    printf '%s  %s\n' "$(sha256_file "$file")" "$relative" >>"$output"
  done < <(find "$root" -type f -print | LC_ALL=C sort)
}

# Remote readback deliberately does not use huggingface_hub or credentials.
# curl >= 8.4 enforces --max-filesize even when Content-Length is absent.
SMALL_FILE_LIMIT=16777216
SMALL_TOTAL_LIMIT=67108864
small_bytes_reserved=0
remote_index=0

safe_relative() {
  [[ "$1" =~ ^[A-Za-z0-9_-][A-Za-z0-9._/-]*$ && "$1" != */ ]] \
    || die "unsafe remote/local relative path: $1"
  case "/$1/" in *'/../'*|*'/./'*|*'//'*) die "non-normalized path: $1" ;; esac
}

# Read ONLY the final HTTP header block (ignoring e.g. proxy CONNECT). Duplicate
# identity headers fail closed rather than selecting whichever one matches.
header_value() {
  awk -v key="$2" '
    /^HTTP\// { count=0; value="" }
    { sub(/\r$/, "") }
    tolower($0) ~ "^" key ":" {
      count++; value=substr($0,index($0,":")+1)
      sub(/^[ \t]+/, "", value); sub(/[ \t]+$/, "", value)
    }
    END { if (count > 1) exit 2; if (count == 1) print value }
  ' "$1"
}

remote_head() {
  local url=$1 expected_revision=$2 headers=$3 status commit
  # Do not follow HEAD redirects: identity must come from the HF resolve endpoint,
  # not a CDN ETag (which may be a multipart hash, not an LFS SHA-256).
  status=$(curl -q --silent --show-error --fail --head \
    --proto '=https' --proto-redir '=https' --connect-timeout 20 --max-time 120 \
    --dump-header "$headers" --output /dev/null --write-out '%{http_code}' "$url") \
    || die "remote HEAD failed: $url"
  case "$status" in 200|301|302|303|307|308) ;; *) die "invalid HEAD status $status: $url" ;; esac
  commit=$(header_value "$headers" x-repo-commit) || die "duplicate remote commit identity: $url"
  [[ "$commit" =~ ^[0-9a-f]{40}$ && "$commit" == "$expected_revision" ]] \
    || die "missing, invalid or mismatched remote commit identity: $url"
}

small_get() {
  local url=$1 output=$2 cap=$3 status
  [[ "$cap" =~ ^[0-9]+$ ]] || die "invalid metadata readback cap: $url"
  ((cap > 0 && cap <= SMALL_FILE_LIMIT)) || die "small metadata exceeds readback cap: $url"
  small_bytes_reserved=$((small_bytes_reserved + cap))
  ((small_bytes_reserved <= SMALL_TOTAL_LIMIT)) || die "total metadata readback cap exceeded"
  # No auth, no curlrc, no HTTP downgrade, at most five HTTPS redirects. New curl
  # enforces the cap while streaming, including chunked responses. Never called
  # for ONNX files, even if a graph is smaller than this limit.
  status=$(curl -q --silent --show-error --fail --location --max-redirs 5 \
    --proto '=https' --proto-redir '=https' --connect-timeout 20 --max-time 120 \
    --max-filesize "$cap" --dump-header "$output.headers" \
    --output "$output" --write-out '%{http_code}' "$url") \
    || die "bounded metadata GET failed: $url"
  [[ "$status" == 200 ]] || die "invalid GET status $status: $url"
  (($(wc -c <"$output") <= cap)) || die "metadata readback exceeded cap: $url"
}

validate_file_table() {
  jq -e '
    type == "object" and length > 0 and
    all(to_entries[];
      (.key | type == "string") and
      (.value.bytes | type == "number" and . > 0 and . == floor and . <= 9007199254740991) and
      (.value.sha256 | type == "string" and test("^[0-9a-f]{64}$")))
  ' "$1" >/dev/null || die "invalid file identity table: $1"
}

verify_remote_file() {
  local root=$1 relative=$2 repository=$3 remote_revision=$4 remote_path=$5
  local expected_bytes=$6 expected_hash=$7 provenance=$8
  local url headers local_hash linked method remote_hash size
  safe_relative "$relative"
  safe_relative "$remote_path"
  require_file "$root" "$relative"
  size=$(wc -c <"$root/$relative" | tr -d '[:space:]')
  [[ "$size" == "$expected_bytes" ]] || die "local size mismatch: $root/$relative"
  local_hash=$(sha256_file "$root/$relative")
  [[ "$local_hash" == "$expected_hash" ]] || die "local SHA-256 mismatch: $root/$relative"
  remote_index=$((remote_index + 1))
  url="https://huggingface.co/$repository/resolve/$remote_revision/$remote_path"
  headers="$report_dir/remote-identity/$remote_index.head"
  remote_head "$url" "$remote_revision" "$headers"
  linked=$(header_value "$headers" x-linked-etag) || die "duplicate LFS identity: $url"
  # HF quotes ETags; never accept weak ETags, SHA-1 or arbitrary strings.
  if [[ "$linked" == \"*\" ]]; then linked=${linked#\"}; linked=${linked%\"}; fi
  if [[ -n "$linked" ]]; then
    [[ "$linked" =~ ^[0-9a-f]{64}$ && "$linked" == "$local_hash" ]] \
      || die "invalid or mismatched LFS SHA-256: $url"
  fi
  if [[ "$relative" == *.onnx || "$relative" == *.onnx_data || "$relative" == *.data ]]; then
    [[ "$linked" =~ ^[0-9a-f]{64}$ ]] || die "graph/data lacks LFS SHA-256 (no GET fallback): $url"
    method=head-lfs-sha256
    remote_hash=$linked
  else
    # Only known metadata extensions are eligible; no bulk binary fallback.
    case "$relative" in *.json|*.md|*.txt|LICENSE|NOTICE) ;; *) die "no small metadata fallback for $relative" ;; esac
    small_get "$url" "$work_dir/readback-$remote_index" "$expected_bytes"
    remote_hash=$(sha256_file "$work_dir/readback-$remote_index")
    [[ "$remote_hash" == "$local_hash" ]] || die "remote metadata SHA-256 mismatch: $url"
    cp "$work_dir/readback-$remote_index.headers" "$report_dir/remote-identity/$remote_index.get"
    method=get-sha256
  fi
  jq -cn --arg local_path "$root/$relative" --arg repository "$repository" \
    --arg revision "$remote_revision" --arg path "$remote_path" --arg url "$url" \
    --arg sha256 "$remote_hash" --arg method "$method" --arg provenance "$provenance" \
    --arg headers "remote-identity/$remote_index.head" --argjson bytes "$size" \
    '{local_path:$local_path,repository:$repository,verified_revision:$revision,path:$path,
      url:$url,sha256:$sha256,bytes:$bytes,method:$method,provenance:$provenance,headers:$headers}' \
    >>"$report_dir/remote-identities.jsonl"
}

verify_table() {
  local root=$1 table=$2 repository=$3 remote_revision=$4 prefix=$5 provenance=$6
  local relative bytes digest
  validate_file_table "$table"
  # Materialize jq output so parse failure cannot disappear in process substitution.
  jq -r 'to_entries[] | [.key, (.value.bytes|tostring), .value.sha256] | @tsv' \
    "$table" >"$work_dir/file-table.tsv"
  while IFS=$'\t' read -r relative bytes digest; do
    verify_remote_file "$root" "$relative" "$repository" "$remote_revision" \
      "$prefix$relative" "$bytes" "$digest" "$provenance"
  done <"$work_dir/file-table.tsv"
}

require_exact_tree() {
  local root=$1 table=$2 manifest=${3:-} relative
  # Symlinks (including directory symlinks) and special files undermine the
  # fixed local-byte inventory. A fresh downloader bundle contains neither.
  [[ -z "$(find "$root" ! -type d ! -type f -print -quit)" ]] \
    || die "bundle contains a symlink or special file: $root"
  while IFS= read -r -d '' relative; do
    relative=${relative#"$root"/}
    safe_relative "$relative"
    [[ -n "$manifest" && "$relative" == "$manifest" ]] && continue
    jq -e --arg path "$relative" 'has($path)' "$table" >/dev/null \
      || die "unlisted local bundle file: $relative"
  done < <(find "$root" -type f -print0)
}

source_url=$DEFAULT_SOURCE
hf_repository=$DEFAULT_HF_REPOSITORY
revision=
hf_revision=
boundary_arg=
v2_arg=
report_arg=
work_arg=
toolchain=$DEFAULT_TOOLCHAIN
keep_work=0
deny_python_exec=0
python_exec_denied=false
python_scope='PATH-name blocking only; not an interpreter sandbox'
python_policy='(version 1) (allow default) (deny process-exec (regex #"/python[^/]*$"))'

while (($#)); do
  case "$1" in
    --source)
      (($# >= 2)) || die "--source requires a value"
      source_url=$2
      shift 2
      ;;
    --rev)
      (($# >= 2)) || die "--rev requires a value"
      revision=$2
      shift 2
      ;;
    --hf-revision)
      (($# >= 2)) || die "--hf-revision requires a value"
      hf_revision=$2
      shift 2
      ;;
    --hf-repository)
      (($# >= 2)) || die "--hf-repository requires a value"
      hf_repository=$2
      shift 2
      ;;
    --bundle-root)
      (($# >= 2)) || die "--bundle-root requires a value"
      boundary_arg=$2
      shift 2
      ;;
    --v2-bundle-root)
      (($# >= 2)) || die "--v2-bundle-root requires a value"
      v2_arg=$2
      shift 2
      ;;
    --report-dir)
      (($# >= 2)) || die "--report-dir requires a value"
      report_arg=$2
      shift 2
      ;;
    --work-dir)
      (($# >= 2)) || die "--work-dir requires a value"
      work_arg=$2
      shift 2
      ;;
    --toolchain)
      (($# >= 2)) || die "--toolchain requires a value"
      toolchain=$2
      shift 2
      ;;
    --keep-work)
      keep_work=1
      shift
      ;;
    --deny-python-exec)
      deny_python_exec=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || die "--rev must be a full lowercase 40-hex commit SHA"
[[ "$hf_revision" =~ ^[0-9a-f]{40}$ ]] || die "--hf-revision must be a full lowercase 40-hex commit SHA"
[[ "$hf_repository" == "$DEFAULT_HF_REPOSITORY" ]] || die "--hf-repository must be $DEFAULT_HF_REPOSITORY"
[[ -n "$boundary_arg" ]] || die "--bundle-root is required"
[[ -n "$v2_arg" ]] || die "--v2-bundle-root is required"
[[ -n "$report_arg" ]] || die "--report-dir is required"
[[ "$boundary_arg" == /* ]] || die "--bundle-root must be absolute"
[[ "$v2_arg" == /* ]] || die "--v2-bundle-root must be absolute"
[[ "$report_arg" == /* ]] || die "--report-dir must be absolute"
[[ "$source_url" == https://* ]] || die "--source must be a remote HTTPS git URL, never a checkout path"
[[ "$source_url" != *$'\n'* && "$source_url" != *'"'* && "$source_url" != *\\* ]] \
  || die "--source contains characters that cannot be represented safely in generated TOML"
[[ "$toolchain" != *$'\n'* && "$toolchain" != *'/'* ]] || die "invalid --toolchain value"

if ((deny_python_exec)); then
  [[ "$(uname -s)" == Darwin ]] || die "--deny-python-exec requires macOS"
  command -v sandbox-exec >/dev/null 2>&1 || die "--deny-python-exec requires sandbox-exec"
fi
command -v jq >/dev/null 2>&1 || die "jq is required"
command -v curl >/dev/null 2>&1 || die "curl is required"
curl -q --version | jq -Re 'select(startswith("curl ")) | split(" ")[1] | split(".") |
  (.[0]|tonumber) > 8 or ((.[0]|tonumber) == 8 and (.[1]|tonumber) >= 4)' >/dev/null \
  || die "curl >= 8.4 is required for streaming metadata size limits"
command -v git >/dev/null 2>&1 || die "git is required"
command -v rustup >/dev/null 2>&1 || die "rustup is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required"
command -v rustc >/dev/null 2>&1 || die "rustc is required"

script_dir=$(canonical_dir "$(dirname "${BASH_SOURCE[0]}")")
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null) \
  || die "script must be run from a git checkout"
repo_root=$(canonical_dir "$repo_root")
boundary_root=$(canonical_dir "$boundary_arg")
v2_root=$(canonical_dir "$v2_arg")
[[ "$boundary_root" != "$v2_root" ]] || die "boundary and v2 bundle roots must differ"

is_within "$boundary_root" "$repo_root" && die "boundary bundle must be outside the git checkout"
is_within "$v2_root" "$repo_root" && die "v2 bundle must be outside the git checkout"

boundary_files=(
  config.json
  tokenizer.json
  tokenizer_config.json
  encoder_config/config.json
  SOURCE_MODEL_CARD.md
  LICENSE
  NOTICE
  export_manifest.json
  encoder.onnx
  classifier.onnx
  boundary_marginals.onnx
  boundary_scorer.onnx
  boundary_explicit_scorer.onnx
  boundary_records.onnx
  boundary_relations.onnx
)
for file in "${boundary_files[@]}"; do
  require_file "$boundary_root" "$file"
done

v2_files=(
  config.json
  tokenizer.json
  tokenizer_config.json
  encoder.onnx
  extractor_padded.onnx
  classifier.onnx
)
for file in "${v2_files[@]}"; do
  require_file "$v2_root" "$file"
done

[[ ! -e "$report_arg" ]] || die "--report-dir must not already exist: $report_arg"
report_parent=$(canonical_dir "$(dirname "$report_arg")")
report_candidate="$report_parent/$(basename "$report_arg")"
is_within "$report_candidate" "$repo_root" && die "report directory must be outside the git checkout"
is_within "$report_candidate" "$boundary_root" && die "report directory must not modify the downloaded boundary bundle"
is_within "$report_candidate" "$v2_root" && die "report directory must not modify the downloaded v2 bundle"
mkdir "$report_candidate"
report_dir=$(canonical_dir "$report_candidate")

created_work=0
if [[ -n "$work_arg" ]]; then
  [[ "$work_arg" == /* ]] || die "--work-dir must be absolute"
  [[ ! -e "$work_arg" ]] || die "--work-dir must not already exist: $work_arg"
  work_parent=$(canonical_dir "$(dirname "$work_arg")")
  work_candidate="$work_parent/$(basename "$work_arg")"
  is_within "$work_candidate" "$repo_root" && die "work directory must be outside the git checkout"
  is_within "$work_candidate" "$boundary_root" && die "work directory must not modify the downloaded boundary bundle"
  is_within "$work_candidate" "$v2_root" && die "work directory must not modify the downloaded v2 bundle"
  mkdir "$work_candidate"
  work_dir=$(canonical_dir "$work_candidate")
else
  work_dir=$(mktemp -d "${TMPDIR:-/tmp}/gliner2-external-consumer.XXXXXX")
  work_dir=$(canonical_dir "$work_dir")
  created_work=1
fi
is_within "$work_dir" "$repo_root" && die "work directory must be outside the git checkout"
is_within "$work_dir" "$boundary_root" && die "work directory must not modify the downloaded boundary bundle"
is_within "$work_dir" "$v2_root" && die "work directory must not modify the downloaded v2 bundle"

run_status=failed
remote_identity_verified=false
python_path_blocked=false
cleanup() {
  local rc=$?
  {
    printf 'status=%s\n' "$run_status"
    printf 'exit_code=%s\n' "$rc"
    printf 'source_sha=%s\n' "$revision"
    printf 'source_url=%s\n' "$source_url"
    printf 'hf_repository=%s\n' "$hf_repository"
    printf 'requested_publication_revision=%s\n' "$hf_revision"
    printf 'remote_identity_verified=%s\n' "$remote_identity_verified"
    printf 'toolchain=%s\n' "$toolchain"
    printf 'python_path_blocked=%s\n' "$python_path_blocked"
    printf 'python_exec_denial_requested=%s\n' "$deny_python_exec"
    printf 'python_named_exec_denied=%s\n' "$python_exec_denied"
    printf 'python_evidence_scope=%s\n' "$python_scope"
    printf 'boundary_bundle=%s\n' "$boundary_root"
    printf 'v2_bundle=%s\n' "$v2_root"
    printf 'work_dir=%s\n' "$work_dir"
  } >"$report_dir/run-summary.txt"
  if ((keep_work == 0)); then
    rm -rf "$work_dir"
  elif ((created_work == 1)); then
    printf 'retained work directory: %s\n' "$work_dir" >&2
  fi
}
trap cleanup EXIT

exec > >(tee -a "$report_dir/process-output.log") 2>&1

printf 'External consumer preflight (structural/runtime smoke; not an accuracy claim)\n'
printf 'source: %s\nrevision: %s\n' "$source_url" "$revision"
printf 'HF publication to verify: %s@%s\n' "$hf_repository" "$hf_revision"
printf 'boundary bundle: %s\nv2 bundle: %s\n' "$boundary_root" "$v2_root"

rustup run "$toolchain" rustc -Vv >"$report_dir/rustc-version.txt"
rustup run "$toolchain" cargo -Vv >"$report_dir/cargo-version.txt"
cat "$report_dir/rustc-version.txt"
cat "$report_dir/cargo-version.txt"

printf '%s\n' "$source_url" >"$report_dir/source-url.txt"
printf '%s\n' "$revision" >"$report_dir/source-sha.txt"
printf '%s\n' "$hf_repository" >"$report_dir/hf-repository.txt"
printf '%s\n' "$hf_revision" >"$report_dir/requested-hf-revision.txt"
git ls-remote "$source_url" >"$report_dir/source-ls-remote.txt"
grep -Eq "^${revision}[[:space:]]" "$report_dir/source-ls-remote.txt" \
  || die "source SHA is not advertised by the remote; push it before running this proof"

mkdir "$report_dir/remote-identity"
: >"$report_dir/remote-identities.jsonl"
curl -q --version >"$report_dir/curl-version.txt"
jq --version >"$report_dir/jq-version.txt"
pins="$report_dir/v2-metadata-pins.json"
cp "$repo_root/docs/checkpoints/v2-metadata-pins.json" "$pins"
sha256_file "$pins" >"$report_dir/v2-metadata-pins.sha256"
jq -e --arg repo "$hf_repository" '
  .schema_version == 1 and .hash_algorithm == "sha256" and
  .hosted_onnx.repository == $repo and
  (.hosted_onnx.verified_revision | test("^[0-9a-f]{40}$")) and
  (.models | keys == ["gliner2-base-v1", "gliner2-large-v1"])
' "$pins" >/dev/null || die "invalid v2 pin contract"

# Derive the boundary publication directory from the manifest identity, not
# the local directory name. hf_revision inside that manifest is the ORIGINAL
# fastino checkpoint revision; it is not the codesoda publication revision.
boundary_model=$(jq -er '.hf_model' "$boundary_root/export_manifest.json")
case "$boundary_model" in
  fastino/gliner2.5-small-v1|fastino/gliner2.5-base-v1|fastino/gliner2.5-multi-v1) ;;
  *) die "unknown boundary profile: $boundary_model" ;;
esac
boundary_prefix=${boundary_model#fastino/}
manifest_url="https://huggingface.co/$hf_repository/resolve/$hf_revision/$boundary_prefix/export_manifest.json"
remote_head "$manifest_url" "$hf_revision" "$report_dir/remote-identity/manifest.head"
small_get "$manifest_url" "$report_dir/remote-boundary-export-manifest.json" 8388608
cmp -s "$boundary_root/export_manifest.json" "$report_dir/remote-boundary-export-manifest.json" \
  || die "boundary manifest bytes do not match the immutable publication"
jq '.files' "$report_dir/remote-boundary-export-manifest.json" >"$work_dir/boundary-files.json"
validate_file_table "$work_dir/boundary-files.json"
for file in "${boundary_files[@]}"; do
  [[ "$file" == export_manifest.json ]] && continue
  jq -e --arg file "$file" 'has($file)' "$work_dir/boundary-files.json" >/dev/null \
    || die "boundary manifest omits required file: $file"
done
jq -e 'has("export_manifest.json") | not' "$work_dir/boundary-files.json" >/dev/null \
  || die "boundary manifest must not list itself"
require_exact_tree "$boundary_root" "$work_dir/boundary-files.json" export_manifest.json
verify_remote_file "$boundary_root" export_manifest.json "$hf_repository" "$hf_revision" \
  "$boundary_prefix/export_manifest.json" \
  "$(wc -c <"$boundary_root/export_manifest.json" | tr -d '[:space:]')" \
  "$(sha256_file "$report_dir/remote-boundary-export-manifest.json")" boundary-publication-manifest
verify_table "$boundary_root" "$work_dir/boundary-files.json" "$hf_repository" \
  "$hf_revision" "$boundary_prefix/" boundary-publication

# v2 may be locally renamed. Its pinned config bytes uniquely select base/large.
v2_model=$(jq -er --arg hash "$(sha256_file "$v2_root/config.json")" '
  [.models | to_entries[] | select(.value.source_metadata["config.json"].sha256 == $hash) | .key] |
  if length == 1 then .[0] else error("v2 config has no unique pin") end
' "$pins") || die "v2 config does not identify a pinned bundle"
v2_source=$(jq -er --arg model "$v2_model" '.models[$model].source_repository' "$pins")
v2_revision=$(jq -er --arg model "$v2_model" '.models[$model].source_revision' "$pins")
[[ "$v2_source" == "fastino/$v2_model" && "$v2_revision" =~ ^[0-9a-f]{40}$ ]] \
  || die "invalid v2 original source identity"
jq --arg model "$v2_model" '.models[$model].source_metadata' "$pins" >"$work_dir/v2-metadata.json"
jq --arg model "$v2_model" '.models[$model].hosted_onnx_files' "$pins" >"$work_dir/v2-graphs.json"
jq -s '.[0] + .[1]' "$work_dir/v2-metadata.json" "$work_dir/v2-graphs.json" >"$work_dir/v2-files.json"
require_exact_tree "$v2_root" "$work_dir/v2-files.json"
# Do not substitute hosted_onnx.verified_revision for the requested publication:
# that pin is a baseline, while HEAD proves the graphs at THIS publication SHA.
verify_table "$v2_root" "$work_dir/v2-graphs.json" "$hf_repository" \
  "$hf_revision" "$v2_model/" v2-publication-graphs
verify_table "$v2_root" "$work_dir/v2-metadata.json" "$v2_source" \
  "$v2_revision" "" v2-original-source-metadata
remote_identity_verified=true
printf 'All bundle bytes bound to checked immutable remote identities.\n'

cp "$boundary_root/export_manifest.json" "$report_dir/boundary-export-manifest.json"
sha256_file "$boundary_root/export_manifest.json" >"$report_dir/boundary-export-manifest.sha256"
printf 'Computing downloaded bundle checksums...\n'
record_tree_checksums "$boundary_root" "$report_dir/boundary-bundle-checksums.sha256"
record_tree_checksums "$v2_root" "$report_dir/v2-bundle-checksums.sha256"

consumer_dir="$work_dir/consumer"
no_python_bin="$work_dir/no-python-bin"
mkdir -p "$consumer_dir/src" "$no_python_bin"
# Discover versioned executables (including python3.12) without invoking them.
# This is a PATH guard, not proof that Python is absent from the machine.
{
  printf '%s\n' python python2 python3 python3.12 pythonw pypy pypy3
  IFS=: read -r -a path_dirs <<<"$PATH"
  for path_dir in "${path_dirs[@]}"; do
    [[ -n "$path_dir" ]] || path_dir=.
    for executable in "$path_dir"/python* "$path_dir"/pypy*; do
      [[ -f "$executable" && -x "$executable" ]] || continue
      command_name=${executable##*/}
      [[ "$command_name" =~ ^(python|pypy)[A-Za-z0-9._-]*$ ]] || continue
      printf '%s\n' "$command_name"
    done
  done
} | LC_ALL=C sort -u >"$report_dir/python-blocked-names.txt"
while IFS= read -r command_name; do
  cat >"$no_python_bin/$command_name" <<'EOF'
#!/bin/sh
echo "error: a Python command name was invoked through the smoke PATH guard" >&2
exit 97
EOF
  chmod +x "$no_python_bin/$command_name"
done <"$report_dir/python-blocked-names.txt"
cp "$script_dir/consumer-smoke/src/main.rs" "$consumer_dir/src/main.rs"
cat >"$consumer_dir/Cargo.toml" <<EOF
[package]
name = "gliner2-external-consumer-smoke"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
gliner2-rs = { git = "$source_url", rev = "$revision" }
EOF

export CARGO_HOME="$work_dir/cargo-home"
export CARGO_TARGET_DIR="$work_dir/target"
export PATH="$no_python_bin:$PATH"
# Probe every installed shadow by PATH, recording the expected denial code.
: >"$report_dir/python-path-preflight.txt"
while IFS= read -r command_name; do
  rc=0
  /bin/sh -c '"$1" --version' sh "$command_name" \
    >>"$report_dir/python-path-preflight.txt" 2>&1 || rc=$?
  printf '%s exit=%s (expected 97)\n' "$command_name" "$rc" >>"$report_dir/python-path-preflight.txt"
  [[ "$rc" == 97 ]] || die "Python PATH guard failed for $command_name"
done <"$report_dir/python-blocked-names.txt"
python_path_blocked=true
if ((deny_python_exec)); then
  printf '%s\n' "$python_policy" >"$report_dir/python-exec-policy.sb"
  [[ -x /usr/bin/python3 ]] || die "absolute Python denial probe requires /usr/bin/python3"
  rc=0
  sandbox-exec -p "$python_policy" /bin/sh -c '/usr/bin/python3 --version' \
    >"$report_dir/python-exec-preflight.txt" 2>&1 || rc=$?
  printf 'exit=%s (expected 126)\n' "$rc" >>"$report_dir/python-exec-preflight.txt"
  [[ "$rc" == 126 ]] || die "sandbox absolute-path Python denial returned $rc, expected 126"
  grep -Fq 'Operation not permitted' "$report_dir/python-exec-preflight.txt" \
    || die "sandbox absolute-path Python denial preflight lacked the expected denial diagnostic"
  python_exec_denied=true
  python_scope='PATH shadows plus macOS process-exec denial for /python[^/]*$ paths; not interpreter absence'
fi
printf '%s\n' "$python_scope" \
  'Discovered PATH names and probes are recorded in python-*-names/preflight evidence.' \
  'Default PATH-only mode does not block absolute paths or later PATH changes.' \
  'Optional macOS policy wraps Cargo generation/build/run and descendants, denying Python-named paths.' \
  'Neither mode proves interpreter absence or blocks renamed/embedded interpreters against a malicious adversary.' \
  'A truly Python-absent CI container is deferred.' >"$report_dir/python-evidence-scope.txt"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
unset ORT_DYLIB_PATH ORT_LIB_LOCATION

run_cargo() (
  # Do not inherit checkout-local .cargo/config.toml through the process cwd.
  cd "$consumer_dir"
  if ((deny_python_exec)); then
    sandbox-exec -p "$python_policy" rustup run "$toolchain" cargo "$@"
  else
    rustup run "$toolchain" cargo "$@"
  fi
)

printf 'Generating Cargo.lock from the pinned git revision in a clean Cargo home...\n'
run_cargo generate-lockfile --manifest-path "$consumer_dir/Cargo.toml"
cp "$consumer_dir/Cargo.lock" "$report_dir/Cargo.lock"
cp "$consumer_dir/Cargo.toml" "$report_dir/Cargo.toml"
cp "$consumer_dir/src/main.rs" "$report_dir/main.rs"

printf 'Building and running the external consumer with its own target directory; Python scope: %s\n' "$python_scope"
run_cargo run \
  --manifest-path "$consumer_dir/Cargo.toml" \
  --release \
  --locked \
  -- "$boundary_root" "$v2_root"

run_status=success
printf 'SUCCESS: external pinned-source consumer completed structural bundle and API smoke checks.\n'
printf 'Evidence report: %s\n' "$report_dir"
