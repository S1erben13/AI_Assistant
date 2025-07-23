from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from models.embedding import TextEmbedder
from models.text_processing.registry import TextNormalizerRegistry


class BaseRecord(ABC):
    def __init__(
        self,
        uid: str,
        text: str,
        normalizer_registry: Optional[TextNormalizerRegistry] = None,
        embedder: Optional[TextEmbedder] = None,
    ):
        self.uid = uid
        self.original_text = text
        self._normalized_text = None
        self.normalizer_registry = (
            normalizer_registry or self._create_default_registry()
        )
        self.embedder = embedder

    @abstractmethod
    def _create_default_registry(self) -> TextNormalizerRegistry:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    def normalize_text(self) -> str:
        """Applies all registered normalizers (with caching)"""
        if self._normalized_text is None:
            self._normalized_text = self.normalizer_registry.normalize(self.original_text)
        return self._normalized_text

    def to_embedding(self) -> np.ndarray:
        if not self.embedder:
            raise ValueError("Embedder not initialized")
        return self.embedder.embed(self.normalize_text())

    @property
    def embedding(self) -> np.ndarray:
        return self.to_embedding()

    @property
    def text(self) -> str:
        """Returns normalized text (lazy normalization)"""
        return self.normalize_text()
