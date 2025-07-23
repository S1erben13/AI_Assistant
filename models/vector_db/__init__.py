from typing import Dict, Type
from abc import ABC, abstractmethod

from models.vector_db.base import VectorDatabase


class VectorDBFactory:
    """A factory class for creating vector database instances based on configuration.

    This class provides a centralized way to create different types of vector databases
    while abstracting the instantiation logic. It supports registration of new database
    types and creates instances based on configuration.

    Example Usage:
        # Register database types (typically done once at startup)
        VectorDBFactory.register_db_type("qdrant", QdrantVectorDB)
        VectorDBFactory.register_db_type("mock", MockVectorDB)

        # Create database instance
        config = {"type": "qdrant", "host": "localhost", "port": 6333}
        db = VectorDBFactory.create(config)
    """

    _db_registry: Dict[str, Type[VectorDatabase]] = {}

    @classmethod
    def register_db_type(cls, db_type: str, db_class: Type[VectorDatabase]):
        """Registers a new vector database type with the factory.

        Args:
            db_type (str): The type identifier for the database (e.g., "qdrant")
            db_class (Type[VectorDatabase]): The class implementing the vector database interface

        Raises:
            ValueError: If the db_type is already registered
        """
        if db_type in cls._db_registry:
            raise ValueError(f"Database type '{db_type}' is already registered")
        cls._db_registry[db_type] = db_class

    @classmethod
    def create(cls, config: Dict) -> VectorDatabase:
        """Creates a vector database instance based on configuration.

        Args:
            config (Dict): Configuration dictionary containing at least a 'type' key

        Returns:
            VectorDatabase: An instance of the requested vector database type

        Raises:
            ValueError: If the specified database type is not registered
        """
        db_type = config.get("type", "qdrant")

        if db_type not in cls._db_registry:
            raise ValueError(f"Unsupported DB type: {db_type}. "
                             f"Available types: {list(cls._db_registry.keys())}")

        return cls._db_registry[db_type](config)
