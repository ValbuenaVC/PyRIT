# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

"""
Contains metadata objects for datasets (i.e. subclasses of SeedDatasetProvider).

SeedDatasetMetadata is the internal schema used to normalize metadata fields
from different sources:
- Remote providers that declare metadata as class attributes
- Local prompt files that store metadata at the top level

SeedDatasetFilter is the user-facing filter schema consumed by
SeedDatasetProvider.get_all_dataset_names().

Size and modality are string literals rather than enums for usability — callers
don't need to import extra types to construct a filter.
"""

# Documented expected values for string-typed metadata fields.
# These are not enforced at runtime but serve as documentation.
SeedDatasetSizeLiteral = Literal["tiny", "small", "medium", "large", "huge"]
"""tiny (<10), small (10-99), medium (100-499), large (500-4999), huge (5000+)"""

SeedDatasetModalityLiteral = Literal["text", "image", "video", "audio"]

SeedDatasetSourceTypeLiteral = Literal["remote", "local"]


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
    Filter object for datasets. Passed to `get_all_dataset_names` in
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
    modalities: Optional[list[str]] = None
    source_types: Optional[list[str]] = None
    load_times: Optional[list[SeedDatasetLoadTime]] = None
    harm_categories: Optional[list[str]] = None


@dataclass(frozen=True)
class SeedDatasetMetadata:
    """
    Internal schema for dataset metadata. Constructed by _parse_metadata()
    implementations on each provider type.
    """

    tags: Optional[set[str]] = None
    size: Optional[str] = None
    modalities: Optional[list[str]] = None
    source_type: Optional[str] = None
    load_time: SeedDatasetLoadTime = SeedDatasetLoadTime.UNINITIALIZED
    harm_categories: Optional[list[str]] = None
