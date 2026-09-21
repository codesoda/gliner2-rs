#!/usr/bin/env bash
# Export small, multi, then base sequentially. No validation, upload, or promotion.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-${script_dir}/env/.venv/bin/python}"

if [[ ! -x "${python}" ]]; then
  printf 'error: pinned export Python is not executable: %s\n' "${python}" >&2
  printf 'create scripts/export/env/.venv as documented in scripts/export/env/README.md\n' >&2
  exit 2
fi

exec "${python}" "${script_dir}/export_bundle_2_5.py" --model all "$@"
