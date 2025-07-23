from models.text_processing.base import TextNormalizer


class TextNormalizerRegistry:
    """A registry for text normalizers that applies multiple normalization steps to text.

    This class maintains a collection of text normalizers and applies them sequentially
    to input text. Each normalizer is only applied if it detects that normalization is needed.

    Attributes:
        _normalizers (List[TextNormalizer]): A list of registered text normalizer instances.
    """

    def __init__(self):
        """Initializes an empty TextNormalizerRegistry with no normalizers."""
        self._normalizers = []

    def register(self, normalizer: TextNormalizer):
        """Registers a new text normalizer to be applied during normalization.

        Args:
            normalizer (TextNormalizer): A text normalizer instance to be added to the registry.
        """
        self._normalizers.append(normalizer)

    def normalize(self, text: str) -> str:
        """Applies all registered normalizers to the input text.

        Each normalizer is only applied if its needs_fixing() method returns True for the text.
        Normalizers are applied in the order they were registered.

        Args:
            text (str): The input text to be normalized.

        Returns:
            str: The normalized text after applying all relevant normalizers.
        """
        for normalizer in self._normalizers:
            text = normalizer.normalize(text)
        return text
