import numpy as np
import pytest
from loguru import logger

from models.embedding import TextEmbedder
from models.records.wiki import WikiRecord
from models.text_processing.base import TextNormalizer
from models.text_processing.normalizers import SpaceNormalizer
from models.text_processing.registry import TextNormalizerRegistry

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <white>TEST</white> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


@pytest.fixture
def embedder():
    """Fixture providing initialized TextEmbedder."""
    logger.info("Initializing TextEmbedder fixture")
    embedder = TextEmbedder()
    embedder.set_model("cointegrated/rubert-tiny2", device="cpu")
    logger.success("TextEmbedder initialized successfully")
    return embedder


@pytest.fixture
def mock_normalizer_registry():
    """Fixture providing TextNormalizerRegistry with mock normalizers."""
    logger.info("Creating mock normalizer registry")

    class MockNormalizer(TextNormalizer):
        def needs_fixing(self, text):
            result = True
            logger.debug(f"MockNormalizer.needs_fixing('{text}') -> {result}")
            return result

        def normalize(self, text):
            result = f"processed({text})"
            logger.debug(f"MockNormalizer.normalize('{text}') -> '{result}'")
            return result

    registry = TextNormalizerRegistry()
    registry.register(MockNormalizer())
    logger.success("Mock normalizer registry created")
    return registry


@pytest.fixture
def upper_normalizer_registry():
    """Fixture providing TextNormalizerRegistry with UpperNormalizer."""
    logger.info("Creating upper normalizer registry")

    class UpperNormalizer(TextNormalizer):
        def needs_fixing(self, text):
            result = text != text.upper()
            logger.debug(f"UpperNormalizer.needs_fixing('{text}') -> {result}")
            return result

        def normalize(self, text):
            result = text.upper()
            logger.debug(f"UpperNormalizer.normalize('{text}') -> '{result}'")
            return result

    registry = TextNormalizerRegistry()
    registry.register(UpperNormalizer())
    logger.success("Upper normalizer registry created")
    return registry


class TestWikiRecord:
    """Tests for WikiRecord functionality."""

    def test_text_normalization(self, embedder):
        """Test integrated text normalization."""
        logger.info("Starting test_text_normalization")
        broken_text = "  Привет   ".encode().decode("latin-1")
        logger.debug(f"Creating record with broken text: '{broken_text}'")

        record = WikiRecord(uid="1", text=broken_text, ru_wiki_pageid=123)
        logger.debug(f"Record created: {record}")

        assert record.original_text == broken_text
        normalized = record.normalize_text()
        logger.debug(f"Normalized text: '{normalized}'")
        assert normalized == "Привет"
        logger.success("test_text_normalization passed")

    def test_custom_normalizers(self, upper_normalizer_registry):
        """Test custom normalizers work correctly."""
        logger.info("Starting test_custom_normalizers")
        test_text = "test"
        logger.debug(f"Creating record with text: '{test_text}'")

        record = WikiRecord(
            uid="3",
            ru_wiki_pageid=789,
            text=test_text,
            normalizer_registry=upper_normalizer_registry,
        )
        logger.debug(f"Record created with custom normalizers: {record}")

        normalized = record.normalize_text()
        logger.debug(f"Normalized text: '{normalized}'")
        assert normalized == "TEST"
        logger.success("test_custom_normalizers passed")

    def test_embedding_generation(self, embedder):
        """Test text embedding generation."""
        logger.info("Starting test_embedding_generation")
        test_text = "Тестовый текст"
        logger.debug(f"Creating record with text: '{test_text}'")

        record = WikiRecord(
            uid="4", ru_wiki_pageid=101, text=test_text, embedder=embedder
        )
        logger.debug(f"Record created with embedder: {record}")

        embedding = record.to_embedding()
        logger.debug(f"Generated embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        logger.success("test_embedding_generation passed")

    def test_embedding_normalization(self, embedder):
        """Test normalization is applied before embedding."""
        logger.info("Starting test_embedding_normalization")
        dirty_text = "  Текст  с  пробелами  "
        clean_text = "Текст с пробелами"
        logger.debug(f"Dirty text: '{dirty_text}', clean text: '{clean_text}'")

        normalizer_registry = TextNormalizerRegistry()
        normalizer_registry.register(SpaceNormalizer())
        logger.debug("SpaceNormalizer registered")

        record = WikiRecord(
            uid="5",
            ru_wiki_pageid=202,
            text=dirty_text,
            embedder=embedder,
            normalizer_registry=normalizer_registry,
        )
        logger.debug(f"Record created: {record}")

        expected_embedding = embedder.embed(clean_text)
        logger.debug("Generated expected embedding")

        actual_embedding = record.to_embedding()
        logger.debug("Generated actual embedding")

        np.testing.assert_array_almost_equal(
            actual_embedding,
            expected_embedding,
            decimal=5
        )
        logger.success("test_embedding_normalization passed")

    @pytest.mark.parametrize(
        "uid,pageid,text",
        [
            ("6", 303, "Test text"),
            ("7", 404, "Another text"),
        ],
    )
    def test_to_dict(self, uid, pageid, text):
        """Test record conversion to dictionary."""
        logger.info(f"Starting test_to_dict with uid={uid}, pageid={pageid}, text='{text}'")
        record = WikiRecord(uid=uid, ru_wiki_pageid=pageid, text=text)
        logger.debug(f"Record created: {record}")

        result = record.to_dict()
        logger.debug(f"Result dict: {result}")

        assert result["uid"] == uid
        assert result["ru_wiki_pageid"] == pageid
        assert result["text"] == text
        assert "normalized_text" in result
        logger.success(f"test_to_dict passed for uid={uid}")