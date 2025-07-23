import uuid
from typing import Dict, List, Set

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, Filter, PointStruct, ScoredPoint,
                                  VectorParams)

from .base import VectorDatabase


class QdrantVectorDB(VectorDatabase):
    def __init__(self, config: Dict):
        client_params = config.get("vector_db", {}).get("client_params", {})
        self.client = QdrantClient(**client_params, check_compatibility=False)
        self.collection = config["vector_db"]["collection_name"]
        self._init_collection()

    def _init_collection(self):
        """Creates a collection if it does not exist."""
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=312, distance=Distance.COSINE
                ),
            )

    def upsert_batch(self, records: List[Dict], embeddings: np.ndarray) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, record["uid"])),
                vector=embedding.tolist(),
                payload={
                    "text": record["text"],
                    "original_id": record["uid"],
                    "wiki_pageid": record.get("ru_wiki_pageid"),
                },
            )
            for record, embedding in zip(records, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Finding the nearest vectors"""
        results: List[ScoredPoint] = self.client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=top_k,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
                "vector": hit.vector,
            }
            for hit in results
        ]

    def search_by_ids(self, ids: List[str]) -> List[Dict]:
        """Search records by their original IDs"""
        results = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[{"key": "original_id", "match": {"any": ids}}]
            ),
            limit=len(ids),
            with_payload=True,
            with_vectors=False
        )

        return [
            {
                "id": hit.id,
                "payload": hit.payload,
                "original_id": hit.payload.get("original_id")
            }
            for hit in results[0]
        ]

    def delete_by_ids(self, ids: List[str]) -> None:
        """Deleting records by ID"""
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[{"key": "original_id", "match": {"value": ids}}]
            ),
        )

    def check_existing_ids(self, uids: List[str]) -> Set[str]:
        """Checks which original IDs are already in the database."""
        existing_records = self.search_by_ids(uids)
        return {record["original_id"] for record in existing_records if record.get("original_id")}
