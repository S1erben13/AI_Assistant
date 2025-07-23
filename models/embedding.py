from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """A class for generating text embeddings using SentenceTransformer models.

    This class provides functionality to load either pre-trained SentenceTransformer models
    by name or use existing SentenceTransformer instances, and generate embeddings for input text.

    The class automatically handles device placement (CPU/GPU) when loading models by name.

    Attributes:
        _model (SentenceTransformer): The underlying sentence transformer model used for embeddings.
    """

    def __init__(
        self, model: Union[str, SentenceTransformer] = None, device: str = None
    ):
        """Initializes the TextEmbedder with an optional model.

        Args:
            model (Union[str, SentenceTransformer], optional): Either a:
                - String specifying a pre-trained SentenceTransformer model name
                - Pre-loaded SentenceTransformer instance
                If None, the model must be set later using set_model().
            device (str, optional): The device to load the model on ('cpu', 'cuda', etc.).
                If None and loading by name, will auto-select GPU if available.
                Ignored if passing a pre-loaded SentenceTransformer instance.
        """
        self._model = None
        if model:
            self.set_model(model, device)

    def set_model(self, model: Union[str, SentenceTransformer], device: str = None):
        """Sets or changes the embedding model used by this instance.

        Args:
            model (Union[str, SentenceTransformer]): Either a:
                - String specifying a pre-trained SentenceTransformer model name
                - Pre-loaded SentenceTransformer instance
            device (str, optional): The device to load the model on ('cpu', 'cuda', etc.).
                If None and loading by name, will auto-select GPU if available.
                Ignored if passing a pre-loaded SentenceTransformer instance.

        Note:
            When loading by model name, this will download the model if not already cached.
        """
        if isinstance(model, str):
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(model, device=device)
        else:
            self._model = model

    def embed(self, text: str) -> np.ndarray:
        """Generates an embedding vector for the input text.

        Args:
            text (str): The input text to embed. For best results, the text should be
                preprocessed according to the model's requirements.

        Returns:
            np.ndarray: A numpy array containing the text embedding vector.

        Raises:
            ValueError: If called before a model is set via constructor or set_model().
        """
        if not self._model:
            raise ValueError("Model not initialized.")
        return self._model.encode(text)
