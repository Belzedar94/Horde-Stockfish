#!/usr/bin/env python3
"""Assemble authenticated Horde-Stockfish native release assets.

This is the native-only Horde adaptation of the Atomic-Stockfish release
manifest contract. Every archive must have a sibling ``.provenance.json``
descriptor. The assembler authenticates the exact four-archive inventory,
copies and re-hashes every byte, then writes the schema-v1 manifest and
``SHA256SUMS`` last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any, Optional, Sequence


MANIFEST_NAME = "horde-stockfish-release-manifest.json"
CHECKSUM_NAME = "SHA256SUMS"
PROVENANCE_SUFFIX = ".provenance.json"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
ARCHITECTURES = ("x86-64-avx2", "x86-64-bmi2")
PLATFORMS = ("linux", "windows")
PROVENANCE_KEYS = {
    "schemaVersion",
    "asset",
    "version",
    "commit",
    "sourceDateEpoch",
    "kind",
    "platform",
    "architecture",
    "toolchain",
    "buildCommand",
    "sha256",
}


class ReleaseContractError(RuntimeError):
    """The candidate bundle violates the frozen native release contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_asset_names(version: str) -> set[str]:
    names: set[str] = set()
    for platform in PLATFORMS:
        extension = "tar.xz" if platform == "linux" else "zip"
        for architecture in ARCHITECTURES:
            names.add(
                f"Horde-Stockfish-{version}-{platform}-{architecture}.{extension}"
            )
    return names


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseContractError("duplicate JSON key in provenance: " + key)
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseContractError(f"invalid provenance {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseContractError("provenance must be one JSON object: " + str(path))
    return value


def _is_regular_unlinked(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _expected_identity(asset_name: str, version: str) -> tuple[str, str]:
    prefix = f"Horde-Stockfish-{version}-"
    if not asset_name.startswith(prefix):
        raise ReleaseContractError("unexpected release asset name: " + asset_name)
    remainder = asset_name[len(prefix) :]
    for platform in PLATFORMS:
        extension = ".tar.xz" if platform == "linux" else ".zip"
        for architecture in ARCHITECTURES:
            if remainder == f"{platform}-{architecture}{extension}":
                return platform, architecture
    raise ReleaseContractError("unexpected release asset name: " + asset_name)


def validate_provenance(
    value: dict[str, Any],
    asset_name: str,
    version: str,
    commit: str,
    source_date_epoch: int,
) -> None:
    if set(value) != PROVENANCE_KEYS:
        missing = sorted(PROVENANCE_KEYS - set(value))
        extra = sorted(set(value) - PROVENANCE_KEYS)
        raise ReleaseContractError(
            f"provenance keys differ for {asset_name} (missing={missing} extra={extra})"
        )
    platform, architecture = _expected_identity(asset_name, version)
    expected = {
        "schemaVersion": 2,
        "asset": asset_name,
        "version": version,
        "commit": commit,
        "sourceDateEpoch": source_date_epoch,
        "kind": "native",
        "platform": platform,
        "architecture": architecture,
    }
    for key, wanted in expected.items():
        if value[key] != wanted:
            raise ReleaseContractError(
                f"{asset_name} provenance {key} mismatch: {value[key]!r} != {wanted!r}"
            )
    for key in ("toolchain",):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ReleaseContractError(f"empty provenance {key} for {asset_name}")
    command = value["buildCommand"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ReleaseContractError("invalid buildCommand for " + asset_name)
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ReleaseContractError("invalid provenance SHA-256 for " + asset_name)


def discover_assets(
    input_root: Path, version: str, commit: str, source_date_epoch: int
) -> list[tuple[Path, dict[str, Any]]]:
    root = input_root.resolve(strict=True)
    candidates = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.name.endswith(PROVENANCE_SUFFIX)
        ),
        key=lambda item: item.name.casefold(),
    )
    if not candidates:
        raise ReleaseContractError("release input contains no assets")

    seen: set[str] = set()
    discovered: list[tuple[Path, dict[str, Any]]] = []
    for asset in candidates:
        name = asset.name
        folded = name.casefold()
        if (
            name in {MANIFEST_NAME, CHECKSUM_NAME}
            or not SAFE_NAME.fullmatch(name)
            or folded in seen
            or not _is_regular_unlinked(asset)
        ):
            raise ReleaseContractError("unsafe or duplicate release asset name: " + name)
        seen.add(folded)
        try:
            asset.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise ReleaseContractError("release asset escapes input root: " + str(asset)) from error

        provenance_path = asset.with_name(name + PROVENANCE_SUFFIX)
        if not _is_regular_unlinked(provenance_path):
            raise ReleaseContractError("missing regular provenance for " + name)
        provenance = _load_json(provenance_path)
        validate_provenance(provenance, name, version, commit, source_date_epoch)
        if sha256(asset) != provenance["sha256"]:
            raise ReleaseContractError("provenance SHA-256 mismatch for " + name)
        discovered.append((asset, provenance))

    expected = expected_asset_names(version)
    actual = {asset.name for asset, _ in discovered}
    if actual != expected:
        raise ReleaseContractError(
            "native release inventory mismatch "
            f"(missing={sorted(expected - actual)} extra={sorted(actual - expected)})"
        )
    orphaned = sorted(
        path.name
        for path in root.rglob("*" + PROVENANCE_SUFFIX)
        if path.is_file()
        and path.name[: -len(PROVENANCE_SUFFIX)].casefold() not in seen
    )
    if orphaned:
        raise ReleaseContractError("orphaned provenance descriptors: " + repr(orphaned))
    return discovered


def _copy_authenticated(source: Path, destination: Path) -> tuple[int, str]:
    before = source.lstat()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(source), flags)
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
    ):
        os.close(descriptor)
        raise ReleaseContractError("release asset changed before copying: " + source.name)

    source_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as reader, destination.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                source_digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    after = source.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    copied_digest = sha256(destination)
    if (
        before_identity != after_identity
        or copied_digest != source_digest.hexdigest()
        or destination.stat().st_size != before.st_size
    ):
        destination.unlink(missing_ok=True)
        raise ReleaseContractError("release asset changed while copying: " + source.name)
    return before.st_size, copied_digest


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def assemble(
    input_root: Path,
    output_dir: Path,
    version: str,
    commit: str,
    source_date_epoch: int,
) -> dict[str, Any]:
    if not SEMVER.fullmatch(version):
        raise ReleaseContractError("release version must be x.y.z")
    commit = commit.lower()
    if not COMMIT.fullmatch(commit):
        raise ReleaseContractError("release commit must be a full lowercase SHA-1")
    if source_date_epoch < 0:
        raise ReleaseContractError("source-date epoch must be non-negative")

    assets = discover_assets(input_root, version, commit, source_date_epoch)
    output = output_dir.resolve()
    if output.exists():
        raise ReleaseContractError("release output already exists: " + str(output))
    output.mkdir(parents=True, exist_ok=False)

    try:
        entries: list[dict[str, Any]] = []
        for source, provenance in assets:
            size, digest = _copy_authenticated(source, output / source.name)
            if digest != provenance["sha256"]:
                raise ReleaseContractError(
                    "release asset changed after provenance authentication: " + source.name
                )
            entries.append(
                {
                    "name": source.name,
                    "bytes": size,
                    "sha256": digest,
                    "provenance": provenance,
                }
            )
        entries.sort(key=lambda item: item["name"])
        manifest: dict[str, Any] = {
            "schemaVersion": 1,
            "project": "Horde-Stockfish",
            "version": version,
            "tag": "v" + version,
            "commit": commit,
            "sourceDateEpoch": source_date_epoch,
            "artifacts": entries,
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        _write_new(output / MANIFEST_NAME, manifest_bytes)

        checksums = [(item["name"], item["sha256"]) for item in entries]
        checksums.append((MANIFEST_NAME, sha256(output / MANIFEST_NAME)))
        checksum_bytes = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(checksums)
        ).encode("ascii")
        _write_new(output / CHECKSUM_NAME, checksum_bytes)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = assemble(
        args.input_root,
        args.output_dir,
        args.version,
        args.commit,
        args.source_date_epoch,
    )
    print(
        f"assembled {len(manifest['artifacts'])} authenticated "
        f"Horde-Stockfish {manifest['version']} assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
