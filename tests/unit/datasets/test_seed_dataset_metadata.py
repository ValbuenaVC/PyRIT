# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for metadata components related to SeedDatasetProvider.
"""

from pyrit.datasets.seed_datasets.seed_metadata import (
    SeedDatasetFilter,
    SeedDatasetLoadTime,
    SeedDatasetMetadata,
)


class TestMetadataLifecycle:
    """
    Test that the metadata object can be created with different
    subsets of values.
    """

    def test_has_no_values(self):
        metadata = SeedDatasetMetadata()
        assert metadata.tags is None
        assert metadata.size is None
        assert metadata.modalities is None
        assert metadata.source_type is None
        assert metadata.load_time == SeedDatasetLoadTime.UNINITIALIZED
        assert metadata.harm_categories is None

    def test_has_some_values(self):
        metadata = SeedDatasetMetadata(tags={"safety"}, size="large")
        assert metadata.tags == {"safety"}
        assert metadata.size == "large"
        assert metadata.modalities is None
        assert metadata.source_type is None
        assert metadata.load_time == SeedDatasetLoadTime.UNINITIALIZED
        assert metadata.harm_categories is None

    def test_has_all_values(self):
        metadata = SeedDatasetMetadata(
            tags={"default", "safety"},
            size="medium",
            modalities=["text", "image"],
            source_type="remote",
            load_time=SeedDatasetLoadTime.FAST,
            harm_categories=["violence", "illegal"],
        )
        assert metadata.tags == {"default", "safety"}
        assert metadata.size == "medium"
        assert len(metadata.modalities) == 2
        assert metadata.source_type == "remote"
        assert metadata.load_time == SeedDatasetLoadTime.FAST
        assert metadata.harm_categories == ["violence", "illegal"]


class TestFilterLifecycle:
    """
    Test that the filter object can be created with different
    subsets of values.
    """

    def test_has_no_values(self):
        f = SeedDatasetFilter()
        assert f.tags is None
        assert f.sizes is None
        assert f.modalities is None
        assert f.source_types is None
        assert f.load_times is None
        assert f.harm_categories is None

    def test_has_some_values(self):
        f = SeedDatasetFilter(sizes=["large"])
        assert f.sizes == ["large"]
        assert f.tags is None
        assert f.modalities is None

    def test_has_all_values(self):
        f = SeedDatasetFilter(
            tags={"default"},
            sizes=["small", "medium"],
            modalities=["text"],
            source_types=["remote"],
            load_times=[SeedDatasetLoadTime.FAST],
            harm_categories=["violence"],
        )
        assert f.tags == {"default"}
        assert len(f.sizes) == 2
        assert f.modalities == ["text"]
        assert f.source_types == ["remote"]
        assert f.load_times == [SeedDatasetLoadTime.FAST]
        assert f.harm_categories == ["violence"]


class TestMetadataProperties:
    """
    Test that the metadata fields populate correctly.
    """

    def test_size_value(self):
        for size in ["tiny", "small", "medium", "large", "huge"]:
            metadata = SeedDatasetMetadata(size=size)
            assert metadata.size == size

    def test_load_time_value(self):
        for lt in SeedDatasetLoadTime:
            metadata = SeedDatasetMetadata(load_time=lt)
            assert metadata.load_time == lt

    def test_source_value(self):
        for source_type in ["remote", "local"]:
            metadata = SeedDatasetMetadata(source_type=source_type)
            assert metadata.source_type == source_type

    def test_modality_value(self):
        for modality in ["text", "image", "video", "audio"]:
            metadata = SeedDatasetMetadata(modalities=[modality])
            assert modality in metadata.modalities

    def test_tags_value(self):
        metadata = SeedDatasetMetadata(tags={"safety", "default", "custom"})
        assert "safety" in metadata.tags
        assert "default" in metadata.tags
        assert "custom" in metadata.tags

    def test_harm_categories_value(self):
        metadata = SeedDatasetMetadata(harm_categories=["violence", "cybercrime"])
        assert "violence" in metadata.harm_categories
        assert "cybercrime" in metadata.harm_categories


class TestMetadataCoercion:
    """
    Test that _coerce_metadata_values correctly normalizes raw YAML
    values into the types expected by SeedDatasetMetadata.
    """

    def test_tags_list_coerced_to_set(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"tags": ["safety", "default"]})
        assert result["tags"] == {"safety", "default"}
        assert isinstance(result["tags"], set)

    def test_tags_string_coerced_to_set(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"tags": "safety"})
        assert result["tags"] == {"safety"}
        assert isinstance(result["tags"], set)

    def test_tags_normalized_lower_strip(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"tags": ["  Safety ", " DEFAULT"]})
        assert result["tags"] == {"safety", "default"}

    def test_size_coerced_to_lowercase_string(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"size": " Large "})
        assert result["size"] == "large"

    def test_source_type_coerced_to_lowercase_string(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"source_type": " Remote "})
        assert result["source_type"] == "remote"

    def test_load_time_coerced_to_enum(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"load_time": "fast"})
        assert result["load_time"] == SeedDatasetLoadTime.FAST

    def test_load_time_normalized_strip_lower(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"load_time": " Slow "})
        assert result["load_time"] == SeedDatasetLoadTime.SLOW

    def test_modalities_list_coerced_lowercase(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"modalities": ["Text", " IMAGE "]})
        assert result["modalities"] == ["text", "image"]

    def test_modalities_string_coerced_to_list(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"modalities": "text"})
        assert result["modalities"] == ["text"]

    def test_harm_categories_list_coerced_lowercase(self):
        result = SeedDatasetMetadata._coerce_metadata_values(
            raw_metadata={"harm_categories": ["Violence", " Cybercrime "]}
        )
        assert result["harm_categories"] == ["violence", "cybercrime"]

    def test_harm_categories_string_coerced_to_list(self):
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"harm_categories": "violence"})
        assert result["harm_categories"] == ["violence"]

    def test_unknown_type_skipped_with_warning(self, caplog):
        """Unexpected types are dropped and logged, not passed through."""
        result = SeedDatasetMetadata._coerce_metadata_values(raw_metadata={"tags": 12345})
        assert "tags" not in result
        assert "Skipping metadata field" in caplog.text


class TestFilterProperties:
    """
    Test that the filter fields populate correctly.
    """

    def test_sizes_values(self):
        f = SeedDatasetFilter(sizes=["small", "large"])
        assert "small" in f.sizes
        assert "large" in f.sizes

    def test_load_times_values(self):
        f = SeedDatasetFilter(load_times=[SeedDatasetLoadTime.FAST, SeedDatasetLoadTime.SLOW])
        assert SeedDatasetLoadTime.FAST in f.load_times
        assert SeedDatasetLoadTime.SLOW in f.load_times

    def test_sources_values(self):
        f = SeedDatasetFilter(source_types=["local", "remote"])
        assert "local" in f.source_types
        assert "remote" in f.source_types

    def test_modalities_values(self):
        f = SeedDatasetFilter(modalities=["text", "image"])
        assert "text" in f.modalities
        assert "image" in f.modalities

    def test_tags_values(self):
        f = SeedDatasetFilter(tags={"safety", "default"})
        assert "safety" in f.tags
        assert "default" in f.tags

    def test_harm_categories_values(self):
        f = SeedDatasetFilter(harm_categories=["violence", "cybercrime"])
        assert "violence" in f.harm_categories
        assert "cybercrime" in f.harm_categories
