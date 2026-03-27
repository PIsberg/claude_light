"""
Tests for embedding model download progress.

Validates that:
- Models are loaded correctly
- Cache detection works
- Progress display is shown for new downloads
- Graceful degradation when tqdm unavailable
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_light.executor import (
    _check_model_cached, _get_model_cache_dir, 
    _load_embedding_model, _MODEL_SIZES
)


class TestModelCaching:
    """Test model cache detection."""

    def test_get_model_cache_dir(self):
        """Test that cache directory is returned."""
        cache_dir = _get_model_cache_dir()
        assert isinstance(cache_dir, Path)
        assert "cache" in str(cache_dir).lower() or "huggingface" in str(cache_dir).lower()

    def test_check_model_cached_nonexistent(self):
        """Test that non-existent models are not cached."""
        # Use a model name that definitely won't exist
        is_cached = _check_model_cached("fake-model-12345-xyz")
        # Should return False for fake models
        assert is_cached is False

    def test_model_sizes_defined(self):
        """Test that common models have sizes defined."""
        assert "all-MiniLM-L6-v2" in _MODEL_SIZES
        assert "all-mpnet-base-v2" in _MODEL_SIZES
        assert "nomic-ai/nomic-embed-text-v1.5" in _MODEL_SIZES
        
        # Check sizes are reasonable (positive integers)
        for model, size in _MODEL_SIZES.items():
            assert isinstance(size, int)
            assert size > 0


class TestModelLoading:
    """Test model loading with progress."""

    def test_load_embedding_model_quiet_mode(self):
        """Test loading model in quiet mode."""
        with patch('claude_light.executor.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            
            result = _load_embedding_model("all-MiniLM-L6-v2", quiet=True)
            
            # Should call SentenceTransformer
            mock_st.assert_called_once()
            assert result == mock_model

    def test_load_embedding_model_cached(self):
        """Test that cached models load without progress display."""
        with patch('claude_light.executor._check_model_cached', return_value=True):
            with patch('claude_light.executor.SentenceTransformer') as mock_st:
                mock_model = MagicMock()
                mock_st.return_value = mock_model
                
                result = _load_embedding_model("all-MiniLM-L6-v2", quiet=False)
                
                # Should load without progress (just call SentenceTransformer)
                mock_st.assert_called_once()
                assert result == mock_model

    def test_load_embedding_model_with_tqdm(self):
        """Test loading with tqdm progress display."""
        with patch('claude_light.executor._check_model_cached', return_value=False):
            with patch('claude_light.executor._TQDM_AVAILABLE', True):
                with patch('claude_light.executor.tqdm') as mock_tqdm:
                    mock_progress = MagicMock()
                    mock_tqdm.return_value.__enter__.return_value = mock_progress
                    
                    with patch('claude_light.executor.SentenceTransformer') as mock_st:
                        mock_model = MagicMock()
                        mock_st.return_value = mock_model
                        
                        result = _load_embedding_model("all-MiniLM-L6-v2", quiet=False)
                        
                        # Should use tqdm
                        mock_tqdm.assert_called_once()
                        # Should update progress to 100
                        mock_progress.update.assert_called()
                        assert result == mock_model

    def test_load_embedding_model_without_tqdm(self):
        """Test loading without tqdm (fallback)."""
        with patch('claude_light.executor._check_model_cached', return_value=False):
            with patch('claude_light.executor._TQDM_AVAILABLE', False):
                with patch('claude_light.executor.SentenceTransformer') as mock_st:
                    mock_model = MagicMock()
                    mock_st.return_value = mock_model
                    
                    result = _load_embedding_model("all-MiniLM-L6-v2", quiet=False)
                    
                    # Should still load successfully without tqdm
                    mock_st.assert_called_once()
                    assert result == mock_model

    def test_load_embedding_model_different_models(self):
        """Test loading different embedding models."""
        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "nomic-ai/nomic-embed-text-v1.5"
        ]
        
        for model_name in models:
            with patch('claude_light.executor._check_model_cached', return_value=True):
                with patch('claude_light.executor.SentenceTransformer') as mock_st:
                    mock_model = MagicMock()
                    mock_st.return_value = mock_model
                    
                    result = _load_embedding_model(model_name, quiet=True)
                    
                    # Should successfully load each model
                    assert result == mock_model


class TestProgressDisplay:
    """Test progress output messages."""

    def test_model_size_information(self):
        """Test that model sizes are used in messages."""
        # All common models should have defined sizes
        for model in _MODEL_SIZES:
            size = _MODEL_SIZES[model]
            assert size > 0
            assert isinstance(size, int)

    def test_download_message_structure(self):
        """Test download messages are properly formatted."""
        with patch('claude_light.executor._check_model_cached', return_value=False):
            with patch('claude_light.executor._TQDM_AVAILABLE', False):
                with patch('builtins.print') as mock_print:
                    with patch('claude_light.executor.SentenceTransformer'):
                        _load_embedding_model("all-MiniLM-L6-v2", quiet=False)
                    
                    # Should have printed something about download
                    assert mock_print.called


class TestAutoTuneIntegration:
    """Test integration with auto_tune function."""

    def test_auto_tune_uses_new_loader(self):
        """Test that auto_tune uses the new model loading function."""
        from claude_light.executor import auto_tune
        import claude_light.state as state
        
        mock_files = [Mock(spec=Path) for _ in range(50)]
        for f in mock_files:
            f.exists.return_value = True
            f.stat.return_value.st_size = 1000
        
        with patch('claude_light.executor._load_embedding_model') as mock_loader:
            mock_embedder = MagicMock()
            mock_loader.return_value = mock_embedder
            
            # Reset state
            state.EMBED_MODEL = None
            state.embedder = None
            
            try:
                auto_tune(mock_files, quiet=True)
            except Exception:
                # May fail due to mocking, but we just want to verify loader was called
                pass
            
            # Should have called the loader
            assert mock_loader.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
