from typing import Optional

from models.embedding import TextEmbedder
from models.records.base import BaseRecord
from models.text_processing.normalizers import SpaceNormalizer, EncodeNormalizer
from models.text_processing.registry import TextNormalizerRegistry


class WikiRecord(BaseRecord):
    def __init__(
        self,
        uid: str,
        ru_wiki_pageid: int,
        text: str,
        normalizer_registry: Optional[TextNormalizerRegistry] = None,
        embedder: Optional[TextEmbedder] = None,
    ):
        super().__init__(uid, text, normalizer_registry, embedder)
        self.ru_wiki_pageid = ru_wiki_pageid

    def _create_default_registry(self) -> TextNormalizerRegistry:
        registry = TextNormalizerRegistry()
        registry.register(SpaceNormalizer())
        registry.register(EncodeNormalizer())
        return registry

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "ru_wiki_pageid": self.ru_wiki_pageid,
            "text": self.text,
            "normalized_text": self.text
        }
