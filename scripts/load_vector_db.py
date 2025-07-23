import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Generator
import os

from loguru import logger
from numpy import ndarray

from configs import ConfigLoader
from models.text_processing.registry import TextNormalizerRegistry
from models.vector_db import VectorDBFactory, VectorDatabase
from models.vector_db.qdrant_db import QdrantVectorDB

IS_DOCKER = os.getenv('DOCKER_MODE', 'false').lower() == 'true'

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

class DataBatchProcessor:
    """Processes batches of data into vector database records"""

    def __init__(
            self,
            embedder: Any,
            normalizer_registry: Optional[TextNormalizerRegistry] = None,
            record_class: Optional[Any] = None
    ):
        """
        Args:
            embedder: Text embedding model
            normalizer_registry: Optional custom text normalizers
            record_class: Class for creating records (must implement BaseRecord interface)
        """
        self.embedder = embedder
        self.normalizer_registry = normalizer_registry
        self.record_class = record_class

    def process_batch(self, batch: List[Dict]) -> tuple[list[dict], list[ndarray]]:
        """Converts batch of raw data into database-ready format"""
        records = [
            self.record_class(
                uid=str(item["uid"]),
                text=str(item["text"]),
                normalizer_registry=self.normalizer_registry,
                embedder=self.embedder,
                **self._get_extra_record_fields(item)
            )
            for item in batch
        ]

        embeddings = [record.to_embedding() for record in records]
        return [record.to_dict() for record in records], embeddings

    def _get_extra_record_fields(self, item: Dict) -> Dict:
        """Extracts additional fields for record constructor"""
        return {
            'ru_wiki_pageid': int(item["ru_wiki_pageid"])
        } if 'ru_wiki_pageid' in item else {}


class VectorDBLoader:
    """Orchestrates the complete vector database loading process"""

    def __init__(
            self,
            config_loader: ConfigLoader,
            record_class: Any,
            embedder_class: Any,
            db_factory: Any = VectorDBFactory
    ):
        """
        Args:
            config_loader: Loads configuration files
            record_class: Record class implementing BaseRecord interface
            embedder_class: Text embedder class
            db_factory: Vector DB factory (default: VectorDBFactory)
        """
        self.config_loader = config_loader
        self.record_class = record_class
        self.embedder_class = embedder_class
        self.db_factory = db_factory

    def load_data_to_db(self, config_name: str = "embedding"):
        """Main pipeline execution method"""
        logger.info(f"Starting database loading process with config: {config_name}")

        try:
            config = self.config_loader.load(config_name)
            logger.success("Configuration loaded successfully")

            db = self._init_database(config)
            processor = self._init_processor(config)
            data = self._load_data(config)

            self._process_in_batches(data, config, processor, db)
            logger.success("Database loading completed successfully")
        except Exception as e:
            logger.error(f"Error during database loading: {str(e)}")
            raise

    def _init_database(self, config: Dict) -> Any:
        """Initializes vector database"""
        try:
            self.db_factory.register_db_type("qdrant", QdrantVectorDB)
            db = self.db_factory.create(config)
            logger.info(f"Initialized database: {type(db).__name__}")
            return db
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def _init_processor(self, config: Dict) -> DataBatchProcessor:
        """Creates data processor with embedder"""
        try:
            embedder = self.embedder_class(model=config["embedding"]["model_name"])
            logger.info(f"Initialized embedder with model: {config['embedding']['model_name']}")
            return DataBatchProcessor(embedder=embedder, record_class=self.record_class)
        except Exception as e:
            logger.error(f"Failed to initialize processor: {str(e)}")
            raise

    def _load_data(self, config: Dict) -> List[Dict]:
        """Loads and validates input data from JSON file"""
        data_path = Path(__file__).parent.parent / "data" / config["data"]["input_file"]
        logger.info(f"Loading data from: {data_path}")

        if not data_path.exists():
            error_msg = f"Data file not found: {data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                result = data if isinstance(data, list) else [data]
                logger.info(f"Successfully loaded {len(result)} records")
                return result
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in file: {data_path}. Error: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error while loading data: {str(e)}")
            raise

    def _process_in_batches(
            self,
            data: List[Dict],
            config: Dict,
            processor: DataBatchProcessor,
            db: Any
    ):
        """Processes data in batches with detailed logging, handling duplicates both in DB and within the batch."""
        batch_size = config["data"]["batch_size"]
        total_batches = (len(data) + batch_size - 1) // batch_size
        auto_confirm_duplicates = config["data"].get("auto_confirm_duplicates", False)

        logger.info(
            f"Processing {len(data)} records in {total_batches} batches (size: {batch_size})"
        )
        logger.info(f"Duplicate handling: {'Overwrite' if auto_confirm_duplicates else 'Skip'}")

        try:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_uids = [record["uid"] for record in batch]

                seen_uids = set()
                duplicate_in_batch_uids = set()

                for uid in batch_uids:
                    if uid in seen_uids:
                        duplicate_in_batch_uids.add(uid)
                    seen_uids.add(uid)

                if duplicate_in_batch_uids:
                    logger.warning(
                        f"Found {len(duplicate_in_batch_uids)} internal duplicates in batch: "
                        f"{list(duplicate_in_batch_uids)[:5]}{'...' if len(duplicate_in_batch_uids) > 5 else ''}"
                    )
                    if not auto_confirm_duplicates:
                        unique_records = []
                        seen = set()
                        for record in batch:
                            if record["uid"] not in seen:
                                unique_records.append(record)
                                seen.add(record["uid"])
                        batch = unique_records
                        logger.info(f"Filtered internal duplicates. Processing {len(batch)} unique records.")

                if batch:
                    batch_uids = [record["uid"] for record in batch]
                    existing_uids = db.check_existing_ids(batch_uids)

                    if existing_uids:
                        logger.warning(
                            f"Found {len(existing_uids)} duplicates in DB: "
                            f"{list(existing_uids)[:5]}{'...' if len(existing_uids) > 5 else ''}"
                        )
                        if not auto_confirm_duplicates:
                            batch = [record for record in batch if record["uid"] not in existing_uids]
                            logger.info(f"Filtered DB duplicates. Processing {len(batch)} records.")

                if batch:
                    records, embeddings = processor.process_batch(batch)
                    db.upsert_batch(records, embeddings)
                    logger.debug(
                        f"Batch {i // batch_size + 1}/{total_batches} processed - "
                        f"Records: {len(records)}, Embeddings shape: {embeddings[0].shape if embeddings else 'N/A'}"
                    )
                else:
                    logger.info("Skipping batch (all records filtered as duplicates)")

            logger.success(f"Finished processing {len(data)} records in {total_batches} batches")

        except Exception as e:
            logger.error(f"Error in batch {i // batch_size + 1}/{total_batches}: {str(e)}")
            raise


def main():
    """Entry point for script execution"""
    from configs import ConfigLoader
    from models.embedding import TextEmbedder
    from models.records.wiki import WikiRecord

    logger.add(
        "vector_db_loader.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    logger.info("Starting VectorDBLoader application")

    try:
        loader = VectorDBLoader(
            config_loader=ConfigLoader(),
            record_class=WikiRecord,
            embedder_class=TextEmbedder
        )

        test_mode = True
        config_name = "test_embedding" if test_mode else "embedding"
        config_name += "_docker" if IS_DOCKER else ''

        logger.info(f"Running in {'test' if test_mode else 'production'} mode")
        logger.info(f"Docker mode: {IS_DOCKER}")
        logger.info(f"Final config name: {config_name}")

        loader.load_data_to_db(config_name)
    except Exception as e:
        logger.critical(f"Application failed: {str(e)}")
        raise
    finally:
        logger.info("Application shutdown")


if __name__ == "__main__":
    main()