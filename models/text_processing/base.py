from abc import ABC, abstractmethod


class TextNormalizer(ABC):

    @abstractmethod
    def normalize(self, text: str) -> str:
        pass

    @abstractmethod
    def needs_fixing(self, text: str) -> bool:
        pass
