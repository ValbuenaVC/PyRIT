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
