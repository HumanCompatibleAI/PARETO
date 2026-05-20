#!/usr/bin/env python3
"""Validate that Croissant metadata matches the package files on disk."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"

IGNORED_EXACT = {
    "croissant.json",
    "requirements-validation.txt",
}
IGNORED_PREFIXES = (
    ".git/",
    ".github/",
    "scripts/",
)
IGNORED_SUFFIXES = (
    ".pyc",
)


@dataclass(frozen=True)
class IntegrityError:
    category: str
    message: str


@dataclass
class Metrics:
    schema_validated: bool = False
    distribution_entries: int = 0
    distribution_file_objects: int = 0
    described_files: int = 0
    package_files_checked: int = 0
    checked_bytes: int = 0
    errors_by_category: dict[str, int] = field(default_factory=dict)

    def add_error(self, error: IntegrityError) -> None:
        self.errors_by_category[error.category] = self.errors_by_category.get(error.category, 0) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset package root")
    parser.add_argument(
        "--croissant",
        required=True,
        type=Path,
        help="Croissant metadata path, relative to --root or absolute",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip mlcroissant schema validation; integrity checks still run",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="Print every checked file")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def resolve_root(path: Path) -> Path:
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"package root does not exist: {path}")
    if not root.is_dir():
        raise NotADirectoryError(f"package root is not a directory: {path}")
    return root


def resolve_croissant(root: Path, croissant: Path) -> tuple[Path, str]:
    path = croissant if croissant.is_absolute() else root / croissant
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing Croissant file: {croissant}")
    if not path.is_file():
        raise FileNotFoundError(f"Croissant path is not a file: {croissant}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Croissant file must be inside package root: {path}") from exc
    return path, relative


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path.name} is not valid JSON: line {exc.lineno} column {exc.colno}"
        ) from exc


def validate_schema(data: Any) -> None:
    try:
        import mlcroissant as mlc
    except ImportError as exc:
        raise RuntimeError(
            "mlcroissant is not installed; install pinned dependencies with "
            "`python -m pip install -r requirements-validation.txt`"
        ) from exc

    try:
        mlc.Dataset(jsonld=data)
    except Exception as exc:  # mlcroissant raises package-specific validation errors.
        raise RuntimeError(f"mlcroissant validation failed: {exc}") from exc


def type_names(value: Any) -> set[str]:
    values = value if isinstance(value, list) else [value]
    names = set()
    for item in values:
        if isinstance(item, str):
            names.add(item.split(":")[-1])
    return names


def extract_distribution(data: Any) -> list[Any]:
    if not isinstance(data, dict):
        raise ValueError("Croissant JSON must be a JSON object")
    distribution = data.get("distribution")
    if not isinstance(distribution, list) or not distribution:
        raise ValueError("croissant.json distribution must be a nonempty list")
    return distribution


def add_error(errors: list[IntegrityError], metrics: Metrics, category: str, message: str) -> None:
    error = IntegrityError(category=category, message=message)
    errors.append(error)
    metrics.add_error(error)


def validate_content_url(root: Path, content_url: Any) -> tuple[str | None, Path | None, str | None]:
    if not isinstance(content_url, str):
        return None, None, "contentUrl must be a string"
    if not content_url:
        return None, None, "contentUrl must not be empty"
    if "\\" in content_url:
        return None, None, f"contentUrl must use forward slashes: {content_url}"
    if "//" in content_url:
        return None, None, f"contentUrl must not contain empty path segments: {content_url}"

    parsed = urllib.parse.urlparse(content_url)
    if parsed.scheme or parsed.netloc:
        return None, None, f"contentUrl must be local and relative, not a URL: {content_url}"

    raw_parts = content_url.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        return None, None, f"contentUrl contains an unsafe path segment: {content_url}"

    posix = PurePosixPath(content_url)
    if posix.is_absolute():
        return None, None, f"contentUrl must be relative: {content_url}"

    local = root.joinpath(*posix.parts)
    resolved_root = root.resolve()
    resolved_local = local.resolve(strict=False)
    try:
        resolved_local.relative_to(resolved_root)
    except ValueError:
        return None, None, f"contentUrl escapes package root: {content_url}"

    return posix.as_posix(), local, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        prefix = f.read(len(LFS_POINTER_PREFIX))
    return prefix == LFS_POINTER_PREFIX


def is_ignored_package_path(relative_posix: str, croissant_relative: str) -> bool:
    path = PurePosixPath(relative_posix)
    if relative_posix == croissant_relative:
        return True
    if relative_posix in IGNORED_EXACT:
        return True
    if any(relative_posix.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True
    if any(relative_posix.endswith(suffix) for suffix in IGNORED_SUFFIXES):
        return True
    if path.name == ".DS_Store":
        return True
    if "__pycache__" in path.parts:
        return True
    return False


def validate_file_objects(
    root: Path,
    distribution: list[Any],
    errors: list[IntegrityError],
    metrics: Metrics,
    verbose: bool,
) -> set[str]:
    described: set[str] = set()
    content_urls_by_lower: dict[str, str] = {}
    ids: dict[str, int] = {}
    names: dict[str, int] = {}

    metrics.distribution_entries = len(distribution)

    for index, item in enumerate(distribution):
        label = f"distribution[{index}]"
        if not isinstance(item, dict):
            add_error(errors, metrics, "MALFORMED_DISTRIBUTION", f"{label} must be an object")
            continue

        if "FileObject" not in type_names(item.get("@type")):
            add_error(
                errors,
                metrics,
                "UNSUPPORTED_DISTRIBUTION_TYPE",
                f"unsupported distribution entry type at {label}: {item.get('@type')!r}",
            )
            continue
        metrics.distribution_file_objects += 1

        file_id = item.get("@id")
        if not isinstance(file_id, str) or not file_id:
            add_error(errors, metrics, "MALFORMED_FILE_OBJECT", f"{label} missing string @id")
        elif file_id in ids:
            add_error(errors, metrics, "DUPLICATE_ID", f"duplicate FileObject @id: {file_id}")
        else:
            ids[file_id] = index

        name = item.get("name")
        if not isinstance(name, str) or not name:
            add_error(errors, metrics, "MALFORMED_FILE_OBJECT", f"{label} missing string name")
        elif name in names:
            add_error(errors, metrics, "DUPLICATE_NAME", f"duplicate FileObject name: {name}")
        else:
            names[name] = index

        content_url = item.get("contentUrl")
        relative, local_path, unsafe_reason = validate_content_url(root, content_url)
        if unsafe_reason is not None:
            add_error(errors, metrics, "UNSAFE_CONTENT_URL", f"{label}: {unsafe_reason}")
            continue
        assert relative is not None
        assert local_path is not None

        if relative in described:
            add_error(errors, metrics, "DUPLICATE_CONTENT_URL", f"duplicate contentUrl: {relative}")
        described.add(relative)

        lower = relative.casefold()
        previous = content_urls_by_lower.get(lower)
        if previous is not None and previous != relative:
            add_error(
                errors,
                metrics,
                "CASE_COLLISION",
                f"case-colliding contentUrl values: {previous} and {relative}",
            )
        else:
            content_urls_by_lower[lower] = relative

        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            add_error(
                errors,
                metrics,
                "MALFORMED_FILE_OBJECT",
                f"{label} has invalid sha256 for {relative}",
            )
            expected_hash = None

        if not local_path.exists():
            add_error(errors, metrics, "MISSING_FILE", f"Croissant references missing file: {relative}")
            continue
        if local_path.is_symlink():
            add_error(errors, metrics, "SYMLINK_NOT_ALLOWED", f"symlinks are not allowed: {relative}")
            continue
        if local_path.is_dir():
            add_error(
                errors,
                metrics,
                "DESCRIBED_DIRECTORY",
                f"contentUrl points to a directory, not a file: {relative}",
            )
            continue
        if not local_path.is_file():
            add_error(errors, metrics, "MISSING_FILE", f"contentUrl is not a regular file: {relative}")
            continue
        if is_lfs_pointer(local_path):
            add_error(
                errors,
                metrics,
                "LFS_POINTER_FILE",
                f"Git LFS pointer files are unsupported in v1: {relative}",
            )
            continue

        if expected_hash is not None:
            actual_hash = sha256_file(local_path)
            metrics.checked_bytes += local_path.stat().st_size
            if verbose:
                print(f"checked {relative}")
            if actual_hash != expected_hash:
                add_error(
                    errors,
                    metrics,
                    "HASH_MISMATCH",
                    (
                        f"sha256 mismatch: {relative}\n"
                        f"expected: {expected_hash}\n"
                        f"actual:   {actual_hash}"
                    ),
                )

    metrics.described_files = len(described)
    return described


def collect_package_files(
    root: Path,
    croissant_relative: str,
    errors: list[IntegrityError],
    metrics: Metrics,
) -> set[str]:
    package_files: set[str] = set()
    package_by_lower: dict[str, str] = {}

    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if is_ignored_package_path(relative, croissant_relative):
            continue

        if path.is_symlink():
            add_error(errors, metrics, "SYMLINK_NOT_ALLOWED", f"symlinks are not allowed: {relative}")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            continue
        if is_lfs_pointer(path):
            add_error(
                errors,
                metrics,
                "LFS_POINTER_FILE",
                f"Git LFS pointer files are unsupported in v1: {relative}",
            )
            continue

        lower = relative.casefold()
        previous = package_by_lower.get(lower)
        if previous is not None and previous != relative:
            add_error(
                errors,
                metrics,
                "CASE_COLLISION",
                f"case-colliding package files: {previous} and {relative}",
            )
        else:
            package_by_lower[lower] = relative
        package_files.add(relative)

    metrics.package_files_checked = len(package_files)
    return package_files


def compare_package_to_distribution(
    package_files: set[str],
    described_files: set[str],
    errors: list[IntegrityError],
    metrics: Metrics,
) -> None:
    for relative in sorted(package_files - described_files):
        add_error(
            errors,
            metrics,
            "UNDESCRIBED_FILE",
            f"repository file is not described in croissant.json: {relative}",
        )


def metrics_dict(metrics: Metrics, errors: list[IntegrityError]) -> dict[str, Any]:
    by_category: dict[str, list[str]] = {}
    for error in errors:
        by_category.setdefault(error.category, []).append(error.message)
    return {
        "schema_validated": metrics.schema_validated,
        "distribution_entries": metrics.distribution_entries,
        "distribution_file_objects": metrics.distribution_file_objects,
        "described_files": metrics.described_files,
        "package_files_checked": metrics.package_files_checked,
        "checked_bytes": metrics.checked_bytes,
        "error_count": len(errors),
        "errors_by_category": metrics.errors_by_category,
        "errors": by_category,
    }


def print_report(metrics: Metrics, errors: list[IntegrityError]) -> None:
    for error in errors:
        print(f"ERROR [{error.category}]: {error.message}", file=sys.stderr)

    print(f"schema_validated {str(metrics.schema_validated).lower()}")
    print(f"distribution_entries {metrics.distribution_entries}")
    print(f"distribution_file_objects {metrics.distribution_file_objects}")
    print(f"described_files {metrics.described_files}")
    print(f"package_files_checked {metrics.package_files_checked}")
    print(f"checked_bytes {metrics.checked_bytes}")
    print(f"error_count {len(errors)}")
    for category in sorted(metrics.errors_by_category):
        print(f"{category.lower()} {metrics.errors_by_category[category]}")


def main() -> int:
    args = parse_args()
    metrics = Metrics()
    errors: list[IntegrityError] = []

    try:
        root = resolve_root(args.root)
        croissant_path, croissant_relative = resolve_croissant(root, args.croissant)
        data = load_json(croissant_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        return fail(str(exc))

    if not args.skip_schema:
        try:
            validate_schema(data)
            metrics.schema_validated = True
        except RuntimeError as exc:
            return fail(str(exc))

    try:
        distribution = extract_distribution(data)
    except ValueError as exc:
        return fail(str(exc))

    described = validate_file_objects(
        root=root,
        distribution=distribution,
        errors=errors,
        metrics=metrics,
        verbose=args.verbose and not args.json,
    )
    package_files = collect_package_files(
        root=root,
        croissant_relative=croissant_relative,
        errors=errors,
        metrics=metrics,
    )
    compare_package_to_distribution(package_files, described, errors, metrics)

    if args.json:
        print(json.dumps(metrics_dict(metrics, errors), indent=2, sort_keys=True))
    else:
        print_report(metrics, errors)

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
