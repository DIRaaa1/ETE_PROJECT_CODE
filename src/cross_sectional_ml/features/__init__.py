"""Feature construction, normalization, and frozen-column projection."""

from .builder import (
    build_daily_features,
    build_de_raw_features,
    build_early_open_features,
    build_m30_features,
    build_o2o_raw_features,
    build_raw_feature_panel,
    prepare_daily_panel,
    prepare_minute_panel,
    safe_div,
)
from .definitions import (
    D1_FEATURE_COLUMNS,
    DE_RAW_FEATURE_COLUMNS,
    EO_FEATURE_COLUMNS,
    M30_FEATURE_COLUMNS,
    O2O_RAW_FEATURE_COLUMNS,
)
from .manifest import (
    FeatureManifest,
    load_feature_manifest,
    project_feature_panel,
    project_protocol_features,
    validate_manifest_pair,
)
from .normalization import (
    legacy_mad3_v1,
    mad3_zscore_matrix,
    standardize_feature_frame_legacy_v1,
)

__all__ = [
    "D1_FEATURE_COLUMNS",
    "DE_RAW_FEATURE_COLUMNS",
    "EO_FEATURE_COLUMNS",
    "M30_FEATURE_COLUMNS",
    "O2O_RAW_FEATURE_COLUMNS",
    "FeatureManifest",
    "build_daily_features",
    "build_de_raw_features",
    "build_early_open_features",
    "build_m30_features",
    "build_o2o_raw_features",
    "build_raw_feature_panel",
    "legacy_mad3_v1",
    "load_feature_manifest",
    "mad3_zscore_matrix",
    "prepare_daily_panel",
    "prepare_minute_panel",
    "project_feature_panel",
    "project_protocol_features",
    "safe_div",
    "standardize_feature_frame_legacy_v1",
    "validate_manifest_pair",
]
