from loguru import logger
from models.text_processing.base import TextNormalizer

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


class SpaceNormalizer(TextNormalizer):
    def needs_fixing(self, text: str) -> bool:
        if not text:
            logger.debug("Empty string, skipping space check")
            return False

        if "  " in text:
            logger.debug(f"Found multiple spaces in text: '{text}'")
            return True

        has_leading_trailing_spaces = text[0] == " " or text[-1] == " "
        if has_leading_trailing_spaces:
            logger.debug(f"Found leading/trailing spaces in text: '{text}'")
            return True

        if any(char.isspace() and char != " " for char in text):
            logger.debug(f"Found non-standard whitespace in text: '{text}'")
            return True

        logger.debug(f"No space issues found in text: '{text}'")
        return False

    def normalize(self, text: str) -> str:
        if not self.needs_fixing(text):
            logger.debug(f"No space normalization needed for: '{text}'")
            return text

        normalized_text = " ".join(text.split())
        logger.info(f"Normalized spaces: '{text}' → '{normalized_text}'")
        return normalized_text


class EncodeNormalizer(TextNormalizer):
    def needs_fixing(self, text: str) -> bool:
        if not text:
            logger.debug("Empty string, skipping encode check")
            return False

        result = any(c in text for c in {"Ð", "Ñ", "â", "Ã", "Å", "¡"})
        logger.debug(f"Encode check for '{text}': needs fixing={result}")
        return result

    def normalize(self, text: str) -> str:
        logger.debug(f"Starting encode normalization for: '{text}'")

        if not self.needs_fixing(text):
            logger.debug("No encoding issues found")
            return text

        try:
            result = text.encode("latin-1").decode("utf-8")
            logger.success(f"Fixed encoding: '{text}' → '{result}'")
            return result
        except (UnicodeError, AttributeError) as e:
            logger.warning(f"Failed to normalize text '{text}': {str(e)}")
            return text