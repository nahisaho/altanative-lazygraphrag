# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for LazySearchConfig."""

import pytest

from graphrag.config.models.lazy_search_config import LazySearchConfig


class TestLazySearchConfig:
    """Tests for LazySearchConfig."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = LazySearchConfig()
        
        assert config.relevance_budget == 500
        assert config.max_depth == 3
        assert config.top_k_chunks == 100
        assert config.breadth_limit == 10
        assert config.relevance_threshold == 5.0
        assert config.sufficient_relevance_count == 50

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = LazySearchConfig(
            relevance_budget=1000,
            max_depth=5,
            top_k_chunks=150,
            relevance_threshold=6.0,
        )
        
        assert config.relevance_budget == 1000
        assert config.max_depth == 5
        assert config.top_k_chunks == 150
        assert config.relevance_threshold == 6.0

    def test_from_preset_z100(self):
        """Test Z100 preset configuration."""
        config = LazySearchConfig.from_preset("z100")
        
        assert config.relevance_budget == 100
        assert config.sufficient_relevance_count == 20
        assert config.top_k_chunks == 50
        assert config.breadth_limit == 5

    def test_from_preset_z500(self):
        """Test Z500 preset configuration."""
        config = LazySearchConfig.from_preset("z500")
        
        assert config.relevance_budget == 500
        assert config.sufficient_relevance_count == 50
        assert config.top_k_chunks == 100
        assert config.breadth_limit == 10

    def test_from_preset_z1500(self):
        """Test Z1500 preset configuration."""
        config = LazySearchConfig.from_preset("z1500")
        
        assert config.relevance_budget == 1500
        assert config.sufficient_relevance_count == 100
        assert config.top_k_chunks == 200
        assert config.breadth_limit == 20

    def test_from_preset_case_insensitive(self):
        """Test preset names are case insensitive."""
        config1 = LazySearchConfig.from_preset("Z100")
        config2 = LazySearchConfig.from_preset("z100")
        
        assert config1.relevance_budget == config2.relevance_budget

    def test_from_preset_invalid(self):
        """Test invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            LazySearchConfig.from_preset("invalid_preset")

    def test_validation_budget_range(self):
        """Test that budget is within valid range."""
        config = LazySearchConfig(relevance_budget=50)
        assert config.relevance_budget == 50
        
        config_high = LazySearchConfig(relevance_budget=5000)
        assert config_high.relevance_budget == 5000

    def test_validation_threshold_range(self):
        """Test relevance threshold is in valid range."""
        config_low = LazySearchConfig(relevance_threshold=0.0)
        config_high = LazySearchConfig(relevance_threshold=10.0)
        
        assert config_low.relevance_threshold == 0.0
        assert config_high.relevance_threshold == 10.0

    def test_model_fields(self):
        """Test model has expected fields."""
        config = LazySearchConfig()
        
        # Check all expected fields exist
        assert hasattr(config, "relevance_budget")
        assert hasattr(config, "max_depth")
        assert hasattr(config, "top_k_chunks")
        assert hasattr(config, "breadth_limit")
        assert hasattr(config, "relevance_threshold")
        assert hasattr(config, "sufficient_relevance_count")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "temperature")

    def test_model_dump(self):
        """Test config can be serialized."""
        config = LazySearchConfig.from_preset("z500")
        
        data = config.model_dump()
        
        assert isinstance(data, dict)
        assert data["relevance_budget"] == 500
        assert data["max_depth"] == 3

    def test_model_copy(self):
        """Test config can be copied with modifications."""
        config = LazySearchConfig.from_preset("z500")
        
        modified = config.model_copy(update={"relevance_budget": 1000})
        
        assert modified.relevance_budget == 1000
        assert config.relevance_budget == 500  # Original unchanged

    def test_get_preset_name(self):
        """Test get_preset_name method."""
        config_z100 = LazySearchConfig.from_preset("z100")
        config_z500 = LazySearchConfig.from_preset("z500")
        config_z1500 = LazySearchConfig.from_preset("z1500")
        
        assert config_z100.get_preset_name() == "z100"
        assert config_z500.get_preset_name() == "z500"
        assert config_z1500.get_preset_name() == "z1500"
