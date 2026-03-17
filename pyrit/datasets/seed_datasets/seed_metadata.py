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

    Exception for load_times, which defaults to UNINITIALIZED.

    By default, filtering is OR-wise across filter categories and OR-wise within
    filter categories.
    """

    # Tags are a top-level set of labels that assist with filtering.
    # The tag "all" will return every discoverable dataset.
    # The tag "default" will return every dataset with an initialized
    # load_time (i.e., SeedDatasetLoadTime != UNINITIALIZED.) or an explicit
    # "default" tag (think of this like a pinned or starred item).
    tags: Optional[set[str]] = None
    sizes: Optional[list[str]] = None
    modalities: Optional[list[PromptDataType]] = None
    source_types: Optional[list[SeedDatasetSourceType]] = None
    load_times: Optional[list[SeedDatasetLoadTime]] = None
    harm_categories: Optional[list[str]] = None


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
