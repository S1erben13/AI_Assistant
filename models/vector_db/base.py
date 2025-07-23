from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set
import numpy as np


class VectorDatabase(ABC):
    """Abstract base class defining the interface for vector database implementations.

    This class serves as a contract for all concrete vector database implementations,
    ensuring consistent behavior across different database backends.
    """

    @abstractmethod
    def upsert_batch(
            self,
            records: List[Dict[str, Any]],
            embeddings: np.ndarray,
    ) -> None:
        """Insert or update multiple records in the database with their embeddings.

        Args:
            records: List of dictionaries containing record data (metadata, identifiers, etc.)
            embeddings: Numpy array of embeddings with shape (num_records, embedding_dimension)

        Note:
            The implementation should handle both new records and updates to existing ones
            based on the record identifiers.
        """
        pass

    @abstractmethod
    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for the most similar vectors in the database.

        Args:
            query_embedding: Embedding vector to use as the search query
            top_k: Number of most similar results to return

        Returns:
            List of dictionaries containing:
            - 'id': The record identifier
            - 'score': Similarity score
            - 'metadata': Any associated metadata (if available)

        Note:
            The similarity metric (cosine, euclidean, etc.) is implementation-dependent.
        """
        pass

    @abstractmethod
    def search_by_ids(self, ids: List[str]) -> List[Dict]:
        """
            Finds and returns objects by their IDs.

            Parameters:
                ids (List[int]): A list of IDs to search for.

            Returns:
                List[Dict]: A list of found objects.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete records from the database by their identifiers.

        Args:
            ids: List of record identifiers to delete

        Raises:
            KeyError: If any of the specified IDs don't exist in the database
            (implementation-specific whether this is raised or silently handled)
        """
        pass

    @abstractmethod
    def check_existing_ids(self, ids: List[str]) -> Set[str]:
        """Checks which UIDs already exist in the database.

        Parameters:
            ids (List[str]): A list of UIDs to check for existence.

        Returns:
            Set[str]: A set of UIDs that already exist in the database.
        """
        pass