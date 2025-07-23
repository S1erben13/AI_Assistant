import pytest
from loguru import logger

from models.text_processing.normalizers import SpaceNormalizer, EncodeNormalizer

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <white>TEST</white> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

class TestSpaceNormalizer:
    @pytest.mark.parametrize(
        "text,should_fix",
        [("  hello   world  ", True), ("normal", False), ("", False), ("   ", True)],
    )
    def test_space_normalization_detection(self, text, should_fix):
        logger.info(f"Testing needs_fixing for: '{text}'")
        result = SpaceNormalizer().needs_fixing(text)
        logger.debug(f"needs_fixing('{text}') -> {result}")
        assert result == should_fix
        logger.success(f"Passed for text: '{text}'")

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("  hello   world  ", "hello world"),
            ("normal", "normal"),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_space_normalization_fixing(self, text, expected):
        logger.info(f"Testing normalize for: '{text}'")
        result = SpaceNormalizer().normalize(text)
        logger.debug(f"normalize('{text}') -> '{result}'")
        assert result == expected
        logger.success(f"Passed for text: '{text}'")

class TestEncodeNormalizer:
    @pytest.fixture
    def normalizer(self):
        logger.debug("Initializing EncodeNormalizer fixture")
        return EncodeNormalizer()

    @pytest.mark.parametrize("broken,fixed", [
        (
            "ÐÑÐ¾Ð¼Ð½ÑÐµ ÑÐ°ÑÑ Ð½Ð° ÑÐ¸Ð¿Ðµ.",
            "Атомные часы на чипе."
        ),
        (
            "ÐÑÐ¾Ð¼Ð½ÑÐµ ÑÐ°ÑÑ â Ð½Ð° Â«ÑÐ¸Ð¿ÐµÂ».",
            "Атомные часы — на «чипе»."
        ),
    ])
    def test_fixes_broken_encoding(self, normalizer, broken, fixed):
        logger.info(f"Testing broken encoding: '{broken}'")
        result = normalizer.normalize(broken)
        logger.debug(f"normalize() returned: '{result}'")
        assert result == fixed
        logger.success(f"Fixed encoding for: '{broken}'")

    @pytest.mark.parametrize("good_text", [
        "Обычный текст в UTF-8",
        "Hello, world!",
        "",
    ])
    def test_ignores_valid_text(self, normalizer, good_text):
        logger.info(f"Testing valid text: '{good_text}'")
        result = normalizer.normalize(good_text)
        logger.debug(f"normalize() returned unchanged: '{result}'")
        assert result == good_text
        logger.success(f"Valid text unchanged: '{good_text}'")

    def test_handles_garbage(self, normalizer):
        garbage = "!@#$%^&*()"
        logger.info(f"Testing garbage input: '{garbage}'")
        result = normalizer.normalize(garbage)
        logger.debug(f"normalize() returned: '{result}'")
        assert result == garbage
        logger.success("Garbage input handled correctly")