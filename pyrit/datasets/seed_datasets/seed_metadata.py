# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

from pyrit.models.literals import PromptDataType

logger = logging.getLogger(__name__)


"""
Contains metadata objects for datasets (i.e. subclasses of SeedDatasetProvider).

SeedDatasetMetadata is the internal schema used to normalize metadata fields
from different sources:
- Remote providers that declare metadata as class attributes
- Local prompt files that store metadata at the top level

SeedDatasetFilter is the user-facing filter schema consumed by
SeedDatasetProvider.get_all_dataset_names_async().

Size and modality are string literals rather than enums for usability — callers
don't need to import extra types to construct a filter.
"""

# Documented expected values for string-typed metadata fields.
# These are not enforced at runtime but serve as documentation.
SeedDatasetSizeCategory = Literal["tiny", "small", "medium", "large", "huge"]
"""tiny (<10), small (10-99), medium (100-499), large (500-4999), huge (5000+)"""

SeedDatasetSourceType = Literal["remote", "local"]


class SeedDatasetLoadTime(Enum):
    """
    Approximate time to load a dataset. Used to skip slow datasets in fast runs.
    """

    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"

    # Default value for datasets whose load time hasn't been measured.
    UNINITIALIZED = "uninitialized"


@dataclass
class SeedDatasetFilter:
    """
    Filter object for datasets. Passed to `get_all_dataset_names_async` in
    SeedDatasetProvider.

    Most fields are optional. None means "don't filter on this axis."

    By default, filtering is AND across categories (all must match) and
    OR within categories (any overlap is sufficient).

    Setting strict_match=True changes within-category behavior to AND
    for set-like fields (tags, harm_categories, modalities).

    Special tag behavior:
    - "all": A magic bypass that returns every discoverable dataset. When "all"
      is present, ALL other filter fields and strict_match are ignored. This
      operates at the get_all_dataset_names_async level — _match_filter is not
      even called for datasets without metadata.
    - "default": Matches datasets that have "default" in their tags or have an
      initialized load_time. With strict_match=True, "default" loses its
      special shortcut behavior and is treated as a normal tag.
    """

    tags: Optional[set[str]] = None
    sizes: Optional[list[str]] = None
    modalities: Optional[list[PromptDataType]] = None
    source_types: Optional[list[SeedDatasetSourceType]] = None
    load_times: Optional[list[SeedDatasetLoadTime]] = None
    harm_categories: Optional[list[str]] = None

    # Setting this to True forces AND-wise filtering within set-like categories.
    # "all" tag still bypasses everything regardless of this flag.
    strict_match: bool = False

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if self.tags and "all" in self.tags and len(self.tags) > 1:
            logger.warning(
                "Filter has 'all' combined with other tags %s. "
                "'all' bypasses all filtering — other tags will be ignored.",
                self.tags - {"all"},
            )
        if self.tags and "all" in self.tags and self.strict_match:
            logger.warning(
                "Filter has 'all' with strict_match=True. 'all' bypasses all filtering — strict_match has no effect."
            )
        if (
            self.tags
            and "all" in self.tags
            and any(
                f is not None
                for f in [self.sizes, self.modalities, self.source_types, self.load_times, self.harm_categories]
            )
        ):
            logger.warning(
                "Filter has 'all' combined with other filter fields. "
                "'all' bypasses all filtering — other fields will be ignored."
            )


@dataclass(frozen=True)
class SeedDatasetMetadata:
    """
    Internal schema for dataset metadata. Constructed by _parse_metadata()
    implementations on each provider type.
    """

    tags: Optional[set[str]] = None
    size: Optional[SeedDatasetSizeCategory] = None
    modalities: Optional[list[PromptDataType]] = None
    source_type: Optional[SeedDatasetSourceType] = None
    load_time: SeedDatasetLoadTime = SeedDatasetLoadTime.UNINITIALIZED
    harm_categories: Optional[list[str]] = None

    @staticmethod
    def _coerce_metadata_values(*, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Convert YAML primitive values into the types expected by SeedDatasetMetadata.

        Applies .lower().strip() normalization to string values for size, modalities,
        source_type, and harm_categories to prevent case/whitespace mismatches.

        Args:
            raw_metadata (dict[str, Any]): Dictionary of field names to raw YAML-parsed values.

        Returns:
            dict[str, Any]: Dictionary with values coerced to the correct types.
        """
        coerced: dict[str, Any] = {}
        for key, value in raw_metadata.items():
            if key == "tags" and isinstance(value, list):
                coerced[key] = {v.strip().lower() if isinstance(v, str) else v for v in value}
            elif key == "tags" and isinstance(value, str):
                coerced[key] = {value.strip().lower()}
            elif key == "size" and isinstance(value, str) or key == "source_type" and isinstance(value, str):
                coerced[key] = value.strip().lower()
            elif key == "load_time" and isinstance(value, str):
                coerced[key] = SeedDatasetLoadTime(value.strip().lower())
            elif key == "modalities" and isinstance(value, list):
                coerced[key] = [v.strip().lower() if isinstance(v, str) else v for v in value]
            elif key == "modalities" and isinstance(value, str):
                coerced[key] = [value.strip().lower()]
            elif key == "harm_categories" and isinstance(value, list):
                coerced[key] = [v.strip().lower() if isinstance(v, str) else v for v in value]
            elif key == "harm_categories" and isinstance(value, str):
                coerced[key] = [value.strip().lower()]
            else:
                # Unexpected type for a metadata field — skip it with a warning
                # rather than passing garbage into SeedDatasetMetadata.
                logger.warning(
                    f"Skipping metadata field '{key}' with unexpected type {type(value).__name__} (value: {value!r})"
                )
        return coerced
