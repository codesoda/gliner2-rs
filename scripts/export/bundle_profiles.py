"""Immutable source profiles for GLiNER2.5 bundle exports.

Checkpoint identity is established from file content.  Snapshot directory names
are deliberately not used as provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from common import (
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    OFFICIAL_BASE_SOURCE_SHA256,
    sha256_file,
)


@dataclass(frozen=True)
class BundleProfile:
    """A pinned model source and its publication bundle name."""

    model_id: str
    revision: str
    bundle_name: str
    source_sha256: Mapping[str, str]


# The small/multi values were captured in
# /tmp/gliner25-work/m7-source-downloads.json after downloading the immutable HF
# revisions.  The base values remain the accepted pins in common.py.
PROFILES: dict[str, BundleProfile] = {
    "small": BundleProfile(
        model_id="fastino/gliner2.5-small-v1",
        revision="f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
        bundle_name="gliner2.5-small-v1",
        source_sha256={
            "config.json": "0b7d9e1401ceeb83e992ec66d2f93bff7e5646428f1b4706ec527cf88f53578a",
            "encoder_config/config.json": "db837d0dc587f5858687ef860c1f400de10f3c3e44f88daef8cbda80d74e4c9c",
            "model.safetensors": "4ee982787ace270d4bf15dbcb28ced38e0aa201372347114ceedd6336055de2b",
            "tokenizer.json": "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
            "README.md": "28b3444a02a26fc6e21a60aa9006dec77407a30be1b53534dc58cb0bd8ce5b77",
        },
    ),
    "base": BundleProfile(
        model_id=BASE_MODEL_ID,
        revision=BASE_HF_REVISION,
        bundle_name="gliner2.5-base-v1",
        source_sha256={
            **OFFICIAL_BASE_SOURCE_SHA256,
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
            "README.md": "8389c442cade57a25cb8f5f642e73a84651e4ec481ea505a132de259380f9d99",
        },
    ),
    "multi": BundleProfile(
        model_id="fastino/gliner2.5-multi-v1",
        revision="235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        bundle_name="gliner2.5-multi-v1",
        source_sha256={
            "config.json": "8b59a0f426a65859c89cd1ea850c3529c09aa3be3a6fafd8eddfdd17b1bf0146",
            "encoder_config/config.json": "fa4f9ef2903b5369ab172333aae4574e6a476511d7465845cf59f8360ee18716",
            "model.safetensors": "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395",
            "tokenizer.json": "c62446df87ae18ec98b133f8f84fc449a07cc89bbf8ef192a4cb5f9c53777a7a",
            "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
            "README.md": "bc37e83e99dd980241d05b693b07ed82c5f871710159c0c40d1bb3f4b14bc1bd",
        },
    ),
}


def verify_source(profile_name: str, model_dir: str | Path) -> dict[str, str]:
    """Verify and return every pinned source-file hash for ``profile_name``.

    This is intentionally fail-closed: missing files, non-files, and the first
    content mismatch all reject the source before model loading or export.
    """

    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown GLiNER2.5 profile {profile_name!r}; expected one of {choices}"
        ) from exc

    source = Path(model_dir).expanduser()
    verified: dict[str, str] = {}
    errors: list[str] = []
    for relative_name, expected in profile.source_sha256.items():
        path = source / relative_name
        if not path.is_file():
            errors.append(f"{relative_name}: missing regular file")
            continue
        actual = sha256_file(path)
        if actual != expected:
            errors.append(f"{relative_name}: expected sha256 {expected}, got {actual}")
            continue
        verified[relative_name] = actual

    if errors:
        raise RuntimeError(
            f"source verification failed for {profile.model_id}@{profile.revision}: "
            + "; ".join(errors)
        )
    return verified
