"""Frozen 560/602 feature manifests and ordered panel projection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import pandas as pd

Protocol = Literal["o2o", "de"]
EXPECTED_FEATURE_COUNTS: dict[str, int] = {"o2o": 560, "de": 602}
RESOURCE_FILENAMES: dict[str, str] = {
    "o2o": "o2o_features.csv",
    "de": "de_features.csv",
}


def _protocol(value: str) -> Protocol:
    normalized = value.strip().lower()
    if normalized not in EXPECTED_FEATURE_COUNTS:
        raise ValueError("protocol must be 'o2o' or 'de'")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class FeatureManifest:
    """An ordered, protocol-specific retained-feature contract."""

    protocol: Protocol
    columns: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        expected = EXPECTED_FEATURE_COUNTS[self.protocol]
        if len(self.columns) != expected:
            raise ValueError(
                f"{self.protocol} manifest must contain {expected} features; got {len(self.columns)}"
            )
        if any(not column for column in self.columns):
            raise ValueError("manifest contains a blank feature name")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("manifest contains duplicate feature names")


def _read_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return pd.read_csv(source)


def load_feature_manifest(protocol: str, path: str | Path | None = None) -> FeatureManifest:
    """Load the authoritative one-column resource (or an explicitly supplied CSV)."""

    protocol_key = _protocol(protocol)
    if path is None:
        resource = resources.files("cross_sectional_ml.resources").joinpath(RESOURCE_FILENAMES[protocol_key])
        with resources.as_file(resource) as resource_path:
            table = pd.read_csv(resource_path)
        source = f"package:cross_sectional_ml.resources/{RESOURCE_FILENAMES[protocol_key]}"
    else:
        table = _read_csv(path)
        source = str(Path(path))
    if list(table.columns) != ["feature"]:
        raise ValueError(f"manifest must have exactly one column named 'feature'; got {list(table.columns)}")
    if table["feature"].isna().any():
        raise ValueError("manifest contains null feature names")
    columns = tuple(table["feature"].astype(str).str.strip())
    return FeatureManifest(protocol=protocol_key, columns=columns, source=source)


def validate_manifest_pair(o2o: FeatureManifest, de: FeatureManifest) -> None:
    """Verify the dissertation invariant that DE begins with all 560 O2O columns."""

    if o2o.protocol != "o2o" or de.protocol != "de":
        raise ValueError("validate_manifest_pair expects O2O then DE")
    if de.columns[: len(o2o.columns)] != o2o.columns:
        raise ValueError("DE's first 560 features do not exactly match O2O order")


def project_feature_panel(
    frame: pd.DataFrame,
    manifest: FeatureManifest,
    *,
    key_columns: tuple[str, str] = ("Code", "Date"),
) -> pd.DataFrame:
    """Project a standardized panel to keys plus the frozen feature order."""

    required = (*key_columns, *manifest.columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(f"source panel is missing {len(missing)} required columns: {preview}{suffix}")
    if frame.loc[:, list(key_columns)].isna().any(axis=None):
        raise ValueError("source panel contains null keys")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"source panel contains duplicate keys: {key_columns}")
    return frame.loc[:, list(required)].copy()


def project_protocol_features(
    frame: pd.DataFrame,
    protocol: str,
    *,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a protocol manifest and apply its exact ordered projection."""

    return project_feature_panel(frame, load_feature_manifest(protocol, manifest_path))
